use anyhow::{Context as _, Result};
use collections::BTreeMap;
use credentials_provider::CredentialsProvider;
use futures::{AsyncReadExt, FutureExt, StreamExt, future::BoxFuture};
use gpui::{App, AppContext, AsyncApp, Context, Entity, Task};
use http_client::{
    AsyncBody, CustomHeaders, HttpClient, HttpRequestExt, Method, Request as HttpRequest,
    RequestBuilderExt,
};
use language_model::chat_completion::ChatCompletionEventMapper;
use language_model::{
    AuthenticateError, IconOrSvg, LanguageModel, LanguageModelClient, LanguageModelCompletionError,
    LanguageModelCompletionStream, LanguageModelEffortLevel, LanguageModelId, LanguageModelName,
    LanguageModelProvider, LanguageModelProviderId, LanguageModelProviderName,
    LanguageModelProviderState, LanguageModelRequest, LanguageModelToolChoiceSupport,
    ModelRateLimiters, ProviderSettingsView, RateLimiter, SubPageProviderSettings,
    unavailable_error,
};
use open_ai::{
    ResponseStreamEvent,
    responses::{Request as ResponseRequest, StreamEvent as ResponsesStreamEvent, stream_response},
    stream_completion,
};
use serde::Deserialize;
pub use settings::OpenAiCompatibleAutoDiscoverMode as AutoDiscoverMode;
pub use settings::OpenAiReasoningEffort;
use settings::Settings;
use std::sync::Arc;
use ui::{ButtonLike, ElevationIndex, IconButton, IconName, Tooltip, prelude::*};
use ui_input::InputField;

use crate::provider::api_compatible::{ApiCompatibleProviderSettings, ApiCompatibleProviderState};
use crate::provider::open_ai::{OpenAiResponseEventMapper, into_open_ai, into_open_ai_response};
pub use settings::OpenAiCompatibleAvailableModel as AvailableModel;
pub use settings::OpenAiCompatibleModelCapabilities as ModelCapabilities;

const API_KEY_PLACEHOLDER: &str = "000000000000000000000000000000000000000000000000000";

// Context window size applied to auto-discovered models, since the standard
// /v1/models response doesn't include token limits. Users can override
// per-model via `available_models` in settings.
const DEFAULT_MAX_TOKENS: u64 = 128_000;

#[derive(Default, Clone, Debug, PartialEq)]
pub struct OpenAiCompatibleSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
    pub custom_headers: CustomHeaders,
    pub auto_discover: bool,
    pub auto_discover_mode: AutoDiscoverMode,
}

impl ApiCompatibleProviderSettings for OpenAiCompatibleSettings {
    fn api_url(&self) -> &str {
        &self.api_url
    }
}

pub type State = ApiCompatibleProviderState<OpenAiCompatibleSettings>;

/// Holds auto-discovered models and the background fetch task.
/// Separate from `ApiCompatibleProviderState` (which is generic and shared
/// with Anthropic-compatible) so the fetch logic stays OpenAI-specific.
#[derive(Default)]
pub struct FetchState {
    fetched_models: Vec<AvailableModel>,
    fetch_model_task: Option<Task<()>>,
}

impl FetchState {
    fn has_models(&self) -> bool {
        !self.fetched_models.is_empty()
    }

    pub(crate) fn refresh(
        &mut self,
        id: &Arc<str>,
        http_client: &Arc<dyn HttpClient>,
        state: &Entity<State>,
        cx: &mut Context<Self>,
    ) {
        let settings = state.read(cx).settings.clone();
        self.refresh_fetch(id, http_client, state, &settings, cx);
    }

    fn refresh_fetch(
        &mut self,
        id: &Arc<str>,
        http_client: &Arc<dyn HttpClient>,
        state: &Entity<State>,
        settings: &OpenAiCompatibleSettings,
        cx: &mut Context<Self>,
    ) {
        // Cancel any in-flight fetch
        self.fetch_model_task = None;

        if !settings.auto_discover {
            log::debug!("Auto-discovery disabled for provider {id}; clearing fetched models");
            self.fetched_models.clear();
            cx.notify();
            return;
        }

        self.fetched_models.clear();
        cx.notify();

        log::info!(
            "Fetching models from OpenAI-compatible provider {id} at {} (mode: {:?})",
            settings.api_url,
            settings.auto_discover_mode
        );

        let id = id.clone();
        let http_client = http_client.clone();
        let state = state.clone();
        let api_url = settings.api_url.clone();
        let extra_headers = settings.custom_headers.clone();
        let auto_discover_mode = settings.auto_discover_mode;

        self.fetch_model_task = Some(cx.spawn(async move |this, cx| {
            let api_key = state.read_with(cx, |state, _cx| state.api_key_state.key(&api_url));
            log::debug!("Provider {id}: api_key present: {}", api_key.is_some());
            let result = fetch_models(
                http_client.as_ref(),
                &api_url,
                api_key.as_deref(),
                &extra_headers,
                auto_discover_mode,
            )
            .await;

            match result {
                Ok(models) => {
                    log::info!(
                        "Fetched {} models from OpenAI-compatible provider {id} at {api_url}",
                        models.len()
                    );
                    log::debug!("Provider {id} fetched models: {models:?}");
                    this.update(cx, |this, cx| {
                        this.fetched_models = models;
                        cx.notify();
                    })
                    .ok();
                }
                Err(error) => {
                    log::warn!(
                        "Failed to fetch models for OpenAI-compatible provider {id}: {error}"
                    );
                    // Keep existing fetched models on failure (graceful degradation)
                }
            }
        }));
    }
}

pub struct OpenAiCompatibleLanguageModelProvider {
    id: LanguageModelProviderId,
    name: LanguageModelProviderName,
    http_client: Arc<dyn HttpClient>,
    state: Entity<State>,
    request_limiters: ModelRateLimiters,
    fetch_state: Entity<FetchState>,
}

impl OpenAiCompatibleLanguageModelProvider {
    pub fn new(
        id: Arc<str>,
        http_client: Arc<dyn HttpClient>,
        credentials_provider: Arc<dyn CredentialsProvider>,
        cx: &mut App,
    ) -> Self {
        let state = State::new(
            id.clone(),
            credentials_provider,
            |id, cx| {
                crate::AllLanguageModelSettings::get_global(cx)
                    .openai_compatible
                    .get(id)
            },
            cx,
        );

        let fetch_state = cx.new(|cx| {
            let observe_id = id.clone();
            let observe_http_client = http_client.clone();
            let observe_state = state.clone();
            cx.observe(&state, move |this: &mut FetchState, _state, cx| {
                let settings = observe_state.read(cx).settings.clone();
                this.refresh_fetch(
                    &observe_id,
                    &observe_http_client,
                    &observe_state,
                    &settings,
                    cx,
                );
            })
            .detach();

            let mut fetch_state = FetchState::default();
            let settings = state.read(cx).settings.clone();
            fetch_state.refresh_fetch(&id, &http_client, &state, &settings, cx);
            fetch_state
        });

        Self {
            id: id.clone().into(),
            name: id.into(),
            http_client,
            state,
            request_limiters: ModelRateLimiters::default(),
            fetch_state,
        }
    }

    /// Merge auto-discovered models with manual `available_models`.
    /// Manual entries override auto-discovered entries by name.
    fn available_models(&self, cx: &App) -> Vec<AvailableModel> {
        let manual = &self.state.read(cx).settings.available_models;
        let fetched = &self.fetch_state.read(cx).fetched_models;

        if fetched.is_empty() {
            return manual.clone();
        }

        let mut merged: BTreeMap<String, AvailableModel> = BTreeMap::default();

        // Auto-discovered models first
        for model in fetched {
            merged.insert(model.name.clone(), model.clone());
        }

        // Manual overrides take precedence
        for model in manual {
            merged.insert(model.name.clone(), model.clone());
        }

        merged.into_values().collect()
    }

    fn language_model(&self, model: &AvailableModel) -> LanguageModel {
        LanguageModel {
            supports_tools: model.capabilities.tools,
            supports_images: model.capabilities.images,
            tool_choice_support: LanguageModelToolChoiceSupport {
                auto: model.capabilities.tools,
                any: model.capabilities.tools,
                none: true,
            },
            supports_streaming_tools: true,
            supports_thinking: default_thinking_reasoning_effort(model).is_some(),
            supported_effort_levels: supported_thinking_effort_levels(model).into(),
            supports_split_token_display: true,
            max_output_tokens: model.max_output_tokens,
            ..LanguageModel::new(
                LanguageModelId::from(model.name.clone()),
                LanguageModelName::from(
                    model
                        .display_name
                        .clone()
                        .unwrap_or_else(|| model.name.clone()),
                ),
                self.id.clone(),
                self.name.clone(),
                format!("openai/{}", model.name),
                model.max_tokens,
            )
        }
    }

    /// The current configuration of `model`, if this provider still offers it.
    fn config(
        &self,
        model: &LanguageModel,
        cx: &App,
    ) -> Result<AvailableModel, LanguageModelCompletionError> {
        self.available_models(cx)
            .iter()
            .find(|available| available.name == model.id.0.as_ref())
            .cloned()
            .ok_or_else(|| unavailable_error(model))
    }

    fn stream_chat_completion(
        &self,
        request_limiter: &RateLimiter,
        request: open_ai::Request,
        cx: &AsyncApp,
    ) -> BoxFuture<
        'static,
        Result<
            futures::stream::BoxStream<'static, Result<ResponseStreamEvent>>,
            LanguageModelCompletionError,
        >,
    > {
        let http_client = self.http_client.clone();

        let (api_key, api_url, extra_headers) = self.state.read_with(cx, |state, _cx| {
            let api_url = &state.settings.api_url;
            (
                state.api_key_state.key(api_url),
                state.settings.api_url.clone(),
                state.settings.custom_headers.clone(),
            )
        });

        let provider = self.name.clone();
        let future = request_limiter.stream(async move {
            let Some(api_key) = api_key else {
                return Err(LanguageModelCompletionError::NoApiKey { provider });
            };
            let request = stream_completion(
                http_client.as_ref(),
                provider.0.as_str(),
                &api_url,
                &api_key,
                request,
                &extra_headers,
            );
            let response = request.await?;
            Ok(response)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }

    fn stream_response(
        &self,
        request_limiter: &RateLimiter,
        request: ResponseRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<futures::stream::BoxStream<'static, Result<ResponsesStreamEvent>>>>
    {
        let http_client = self.http_client.clone();

        let (api_key, api_url, extra_headers) = self.state.read_with(cx, |state, _cx| {
            let api_url = &state.settings.api_url;
            (
                state.api_key_state.key(api_url),
                state.settings.api_url.clone(),
                state.settings.custom_headers.clone(),
            )
        });

        let provider = self.name.clone();
        let future = request_limiter.stream(async move {
            let Some(api_key) = api_key else {
                return Err(LanguageModelCompletionError::NoApiKey { provider });
            };
            let request = stream_response(
                http_client.as_ref(),
                provider.0.as_str(),
                &api_url,
                &api_key,
                request,
                &extra_headers,
            );
            let response = request.await?;
            Ok(response)
        });

        async move { Ok(future.await?.boxed()) }.boxed()
    }
}

// ── Auto-discovery: fetch models from /v1/models ──────────────────────────

/// Response of `GET /v1/models` (standard OpenAI format).
#[derive(Deserialize)]
struct ListModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    id: String,
}

// ── Model info: llama.cpp router /v1/models ────────────────────────────────

/// Response from llama.cpp router's `GET /v1/models` endpoint.
#[derive(Deserialize)]
struct LlamaCppModelsResponse {
    #[serde(default)]
    data: Vec<LlamaCppModelEntry>,
}

#[derive(Deserialize)]
struct LlamaCppModelEntry {
    id: String,
    #[serde(default)]
    architecture: Option<LlamaCppArchitecture>,
    status: Option<LlamaCppStatus>,
}

#[derive(Default, Deserialize)]
struct LlamaCppArchitecture {
    #[serde(default)]
    input_modalities: Vec<String>,
}

#[derive(Default, Deserialize)]
struct LlamaCppStatus {
    #[serde(default)]
    args: Vec<String>,
}

/// Parse llama.cpp router /v1/models response into AvailableModels.
/// Extracts capabilities from the rich response format.
fn parse_llamacpp_models(body: &str) -> Result<Vec<AvailableModel>> {
    let response: LlamaCppModelsResponse =
        serde_json::from_str(body).context("Unable to parse llama.cpp models response")?;

    let mut models = Vec::new();
    for entry in response.data {
        let supports_images = entry
            .architecture
            .as_ref()
            .map(|arch| arch.input_modalities.iter().any(|m| m == "image"))
            .unwrap_or(false);

        let max_tokens = entry
            .status
            .as_ref()
            .and_then(|status| extract_arg_value(&status.args, "--ctx-size"))
            .unwrap_or(DEFAULT_MAX_TOKENS);

        models.push(AvailableModel {
            name: entry.id.clone(),
            display_name: None,
            max_tokens,
            max_output_tokens: None,
            max_completion_tokens: None,
            reasoning_effort: None,
            capabilities: ModelCapabilities {
                tools: true,
                images: supports_images,
                parallel_tool_calls: false,
                prompt_cache_key: false,
                chat_completions: true,
                interleaved_reasoning: false,
                max_tokens_parameter: false,
            },
        });
    }

    Ok(models)
}

/// Extract the value of a command-line argument from the args array.
fn extract_arg_value(args: &[String], name: &str) -> Option<u64> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == name {
            if let Some(value) = iter.next() {
                return value.parse().ok();
            }
        }
    }
    None
}

// ── Model info: LiteLLM /model/info ────────────────────────────────────────

/// Response of LiteLLM's `GET /v1/model/info` endpoint.
#[derive(Deserialize)]
struct LiteLlmModelInfoResponse {
    #[serde(default)]
    data: Vec<LiteLlmModelInfoEntry>,
}

#[derive(Deserialize)]
struct LiteLlmModelInfoEntry {
    model_name: String,
    #[serde(default)]
    model_info: LiteLlmModelInfo,
}

#[derive(Default, Deserialize)]
struct LiteLlmModelInfo {
    #[serde(default)]
    max_tokens: Option<u64>,
    #[serde(default)]
    max_input_tokens: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<u64>,
    #[serde(default)]
    supports_function_calling: Option<bool>,
    #[serde(default)]
    supports_vision: Option<bool>,
    #[serde(default)]
    supports_reasoning: Option<bool>,
    #[serde(default)]
    supports_prompt_caching: Option<bool>,
}

/// Enrich auto-discovered models with capability/token data from LiteLLM's
/// `/v1/model/info` response. Pure — no I/O; takes the raw JSON body and
/// updates models in place. Models without a matching `/model/info` entry
/// keep their default capabilities.
fn enrich_models_lite_llm(models: &mut [AvailableModel], model_info_body: &str) {
    let Ok(response) = serde_json::from_str::<LiteLlmModelInfoResponse>(model_info_body) else {
        return;
    };
    let info: collections::HashMap<&str, &LiteLlmModelInfo> = response
        .data
        .iter()
        .map(|entry| (entry.model_name.as_str(), &entry.model_info))
        .collect();
    for model in models.iter_mut() {
        let Some(info) = info.get(model.name.as_str()) else {
            continue;
        };
        model.max_tokens = info
            .max_input_tokens
            .or(info.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS);
        model.max_output_tokens = info.max_output_tokens.or(info.max_tokens);
        model.capabilities.tools = info.supports_function_calling.unwrap_or(true);
        model.capabilities.images = info.supports_vision.unwrap_or(false);
        model.capabilities.prompt_cache_key = info.supports_prompt_caching.unwrap_or(false);
        if info.supports_reasoning.unwrap_or(false) {
            model.reasoning_effort = Some(OpenAiReasoningEffort::Medium);
        }
    }
}

/// Fetch the provider's model list from its `/v1/models` endpoint.
///
/// `api_url` already includes the `/v1` prefix (matching the convention used
/// for `/chat/completions`), so `/models` is appended directly. When
/// auto-discover mode is set, a second request fetches capability/token data
/// from a provider-specific endpoint. Discovered models are returned with
/// default capabilities unless enriched; users can override individual
/// models via `available_models` in settings.
async fn fetch_models(
    client: &dyn HttpClient,
    api_url: &str,
    api_key: Option<&str>,
    extra_headers: &CustomHeaders,
    auto_discover_mode: AutoDiscoverMode,
) -> Result<Vec<AvailableModel>> {
    let models_url = format!("{api_url}/models");
    let request = HttpRequest::builder()
        .method(Method::GET)
        .uri(&models_url)
        .header("Accept", "application/json")
        .when_some(api_key, |builder, key| {
            builder.header("Authorization", format!("Bearer {key}"))
        })
        .extra_headers(extra_headers)
        .body(AsyncBody::default())?;

    let mut response = client.send(request).await?;
    let mut body = String::new();
    response.body_mut().read_to_string(&mut body).await?;
    anyhow::ensure!(
        response.status().is_success(),
        "Failed to fetch models: {} {}",
        response.status(),
        body,
    );
    let models_response: ListModelsResponse =
        serde_json::from_str(&body).context("Unable to parse /v1/models response")?;

    let mut models = Vec::new();
    for entry in models_response.data {
        models.push(AvailableModel {
            name: entry.id.clone(),
            display_name: None,
            max_tokens: DEFAULT_MAX_TOKENS,
            max_output_tokens: None,
            max_completion_tokens: None,
            reasoning_effort: None,
            capabilities: ModelCapabilities {
                tools: true,
                images: false,
                parallel_tool_calls: false,
                prompt_cache_key: false,
                chat_completions: true,
                interleaved_reasoning: false,
                max_tokens_parameter: false,
            },
        });
    }

    if auto_discover_mode == AutoDiscoverMode::LlamaCpp {
        // llama.cpp router provides capabilities directly in /v1/models response
        return parse_llamacpp_models(&body);
    }

    if auto_discover_mode == AutoDiscoverMode::LiteLlm {
        let root_url = api_url.strip_suffix("/v1").unwrap_or(api_url);
        let info_url = format!("{root_url}/model/info");
        let info_request = HttpRequest::builder()
            .method(Method::GET)
            .uri(&info_url)
            .header("Accept", "application/json")
            .when_some(api_key, |builder, key| {
                builder.header("Authorization", format!("Bearer {key}"))
            })
            .extra_headers(extra_headers)
            .body(AsyncBody::default())?;

        if let Ok(mut info_response) = client.send(info_request).await {
            if info_response.status().is_success() {
                let mut info_body = String::new();
                if info_response
                    .body_mut()
                    .read_to_string(&mut info_body)
                    .await
                    .is_ok()
                {
                    enrich_models_lite_llm(&mut models, &info_body);
                }
            }
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(models)
}

impl LanguageModelProviderState for OpenAiCompatibleLanguageModelProvider {
    type ObservableEntity = FetchState;

    fn observable_entity(&self) -> Option<Entity<Self::ObservableEntity>> {
        Some(self.fetch_state.clone())
    }
}

impl LanguageModelProvider for OpenAiCompatibleLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelProviderName {
        self.name.clone()
    }

    fn icon(&self) -> IconOrSvg {
        IconOrSvg::Icon(IconName::AiOpenAiCompat)
    }

    fn default_model(&self, cx: &App) -> Option<LanguageModel> {
        self.available_models(cx)
            .first()
            .map(|model| self.language_model(model))
    }

    fn default_fast_model(&self, _cx: &App) -> Option<LanguageModel> {
        None
    }

    fn provided_models(&self, cx: &App) -> Vec<LanguageModel> {
        self.available_models(cx)
            .iter()
            .map(|model| self.language_model(model))
            .collect()
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        // Authenticated if the API key is set AND either manual models exist
        // or auto-discovered models were successfully fetched (meaning the
        // server is reachable and the key works).
        let has_manual = !self.state.read(cx).settings.available_models.is_empty();
        let has_fetched = self.fetch_state.read(cx).has_models();
        let has_key = self.state.read(cx).is_authenticated();
        has_key && (has_manual || has_fetched)
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn settings_view(&self, _cx: &mut App) -> Option<ProviderSettingsView> {
        let state = self.state.clone();
        let fetch_state = self.fetch_state.clone();
        let id: Arc<str> = self.id.0.clone().into();
        let http_client = self.http_client.clone();
        Some(ProviderSettingsView::SubPage(SubPageProviderSettings::new(
            move |window, cx| {
                cx.new(|cx| {
                    OpenAiCompatibleConfigurationView::new(
                        state.clone(),
                        fetch_state.clone(),
                        id.clone(),
                        http_client.clone(),
                        window,
                        cx,
                    )
                })
                .into()
            },
        )))
    }

    fn set_api_key(&self, api_key: Option<String>, cx: &mut App) -> Task<Result<()>> {
        self.state
            .update(cx, |state, cx| state.set_api_key(api_key, cx))
    }
}

/// Configuration view for OpenAI-compatible providers. Renders the API
/// key section, connection status with refresh button (when auto-discover
/// is enabled), and remove provider button.
pub struct OpenAiCompatibleConfigurationView {
    api_key_editor: Entity<InputField>,
    fetch_state: Entity<FetchState>,
    state: Entity<State>,
    id: Arc<str>,
    http_client: Arc<dyn HttpClient>,
    load_credentials_task: Option<Task<()>>,
}

impl OpenAiCompatibleConfigurationView {
    pub fn new(
        state: Entity<State>,
        fetch_state: Entity<FetchState>,
        id: Arc<str>,
        http_client: Arc<dyn HttpClient>,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let api_key_editor = cx.new(|cx| InputField::new(window, cx, API_KEY_PLACEHOLDER));

        cx.observe(&fetch_state, |_, _, cx| cx.notify()).detach();
        cx.observe(&state, |_, _, cx| cx.notify()).detach();

        let load_credentials_task = Some(cx.spawn_in(window, {
            let state = state.clone();
            async move |this, cx| {
                let task = state.update(cx, |state, cx| state.authenticate(cx));
                match task.await {
                    Ok(()) | Err(AuthenticateError::CredentialsNotFound) => {}
                    Err(error) => {
                        log::error!(
                            "Failed to load OpenAI-compatible provider credentials: {error}"
                        );
                    }
                }
                this.update(cx, |this, cx| {
                    this.load_credentials_task = None;
                    cx.notify();
                })
                .ok();
            }
        }));

        Self {
            api_key_editor,
            fetch_state,
            state,
            id,
            http_client,
            load_credentials_task,
        }
    }

    fn refresh(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let state = self.state.clone();
        let fetch_state = self.fetch_state.clone();
        let id = self.id.clone();
        let http_client = self.http_client.clone();

        cx.spawn(async move |_, cx| {
            let auth_task = state.update(cx, |state, cx| state.authenticate(cx));
            let _ = auth_task.await;

            fetch_state.update(cx, |fetch_state, cx| {
                fetch_state.refresh(&id, &http_client, &state, cx);
            });
        })
        .detach();
    }

    fn save_api_key(&mut self, _: &menu::Confirm, window: &mut Window, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx).trim().to_string();
        if api_key.is_empty() {
            return;
        }

        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(Some(api_key), cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn reset_api_key(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        self.api_key_editor
            .update(cx, |input, cx| input.set_text("", window, cx));

        let state = self.state.clone();
        cx.spawn_in(window, async move |_, cx| {
            state
                .update(cx, |state, cx| state.set_api_key(None, cx))
                .await
        })
        .detach_and_log_err(cx);
    }

    fn remove_provider(&mut self, _: &mut Window, cx: &mut Context<Self>) {
        let id = self.id.clone();
        let fs = <dyn fs::Fs>::global(cx);
        settings::update_settings_file(fs, cx, move |settings, _| {
            let Some(language_models) = settings.language_models.as_mut() else {
                return;
            };
            if let Some(providers) = language_models.openai_compatible.as_mut() {
                providers.remove(id.as_ref());
            }
        });
    }

    fn should_render_editor(&self, cx: &Context<Self>) -> bool {
        !self.state.read(cx).is_authenticated()
    }
}

impl Render for OpenAiCompatibleConfigurationView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        if self.load_credentials_task.is_some() {
            return div().child(Label::new("Loading credentials…")).into_any();
        }

        let state = self.state.read(cx);
        let env_var_set = state.api_key_state.is_from_env_var();
        let env_var_name = state.api_key_state.env_var_name();
        let auto_discover = state.settings.auto_discover;
        let has_models = self.fetch_state.read(cx).has_models();

        let api_key_section = if self.should_render_editor(cx) {
            v_flex()
                .on_action(cx.listener(Self::save_api_key))
                .child(Label::new(
                    "To use Zed's agent with an OpenAI-compatible provider, you need to add an API key.",
                ))
                .child(
                    div()
                        .pt(DynamicSpacing::Base04.rems(cx))
                        .child(self.api_key_editor.clone()),
                )
                .child(
                    Label::new(format!(
                        "You can also set the {env_var_name} environment variable and restart Zed.",
                    ))
                    .size(LabelSize::Small)
                    .color(Color::Muted),
                )
                .into_any()
        } else {
            h_flex()
                .mt_1()
                .p_1()
                .justify_between()
                .rounded_md()
                .border_1()
                .border_color(cx.theme().colors().border)
                .bg(cx.theme().colors().background)
                .child(
                    h_flex()
                        .flex_1()
                        .min_w_0()
                        .gap_1()
                        .child(Icon::new(IconName::Check).color(Color::Success))
                        .child(
                            div().w_full().overflow_x_hidden().text_ellipsis().child(Label::new(
                                if env_var_set {
                                    format!("API key set in {env_var_name} environment variable")
                                } else {
                                    format!("API key configured for {}", state.settings.api_url())
                                },
                            )),
                        ),
                )
                .child(
                    h_flex().flex_shrink_0().child(
                        Button::new("reset-api-key", "Reset API Key")
                            .label_size(LabelSize::Small)
                            .start_icon(Icon::new(IconName::Undo).size(IconSize::Small))
                            .layer(ElevationIndex::ModalSurface)
                            .when(env_var_set, |this| {
                                this.tooltip(Tooltip::text(format!(
                                    "To reset your API key, unset the {env_var_name} environment variable.",
                                )))
                            })

                            .on_click(cx.listener(|this, _, window, cx| {
                                this.reset_api_key(window, cx)
                            })),
                    ),
                )
                .into_any()
        };

        v_flex()
            .size_full()
            .gap_4()
            .child(api_key_section)
            .when(auto_discover, |this| {
                this.child(
                    h_flex().w_full().justify_end().child(
                        h_flex()
                            .gap_1()
                            .when(has_models, |this| {
                                this.child(
                                    ButtonLike::new("connected")
                                        .size(ButtonSize::Compact)
                                        .child(
                                            h_flex()
                                                .gap_1()
                                                .child(
                                                    Icon::new(IconName::Check)
                                                        .color(Color::Success),
                                                )
                                                .child(Label::new("Connected")),
                                        )
                                        .child(
                                            IconButton::new("refresh-models", IconName::RotateCcw)
                                                .icon_size(IconSize::Small)
                                                .tooltip(Tooltip::text("Refresh Models"))
                                                .on_click(cx.listener(|this, _, window, cx| {
                                                    this.refresh(window, cx);
                                                })),
                                        ),
                                )
                            })
                            .when(!has_models, |this| {
                                this.child(
                                    Button::new("connect", "Connect")
                                        .style(ButtonStyle::Outlined)
                                        .size(ButtonSize::Compact)
                                        .start_icon(
                                            Icon::new(IconName::PlayOutlined)
                                                .size(IconSize::XSmall),
                                        )
                                        .on_click(cx.listener(|this, _, window, cx| {
                                            this.refresh(window, cx);
                                        })),
                                )
                            }),
                    ),
                )
            })
            .child(
                h_flex().w_full().justify_end().child(
                    Button::new("remove-compatible-provider", "Remove Provider")
                        .style(ButtonStyle::OutlinedGhost)
                        .label_size(LabelSize::Small)
                        .start_icon(
                            Icon::new(IconName::Trash)
                                .size(IconSize::Small)
                                .color(Color::Muted),
                        )
                        .on_click(
                            cx.listener(|this, _, window, cx| this.remove_provider(window, cx)),
                        ),
                ),
            )
            .into_any()
    }
}
impl LanguageModelClient for OpenAiCompatibleLanguageModelProvider {
    fn stream_completion(
        &self,
        model: &LanguageModel,
        mut request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<LanguageModelCompletionStream, LanguageModelCompletionError>>
    {
        let config = match cx.update(|cx| self.config(model, cx)) {
            Ok(config) => config,
            Err(error) => return async move { Err(error) }.boxed(),
        };
        let request_limiter = self.request_limiters.for_model(&model.id);
        // `speed` can leak in from a parent thread's model; this provider never
        // supports fast mode, and arbitrary compatible endpoints reject `service_tier`.
        request.speed = None;

        if config.capabilities.chat_completions {
            let reasoning_effort = chat_completion_reasoning_effort(&request, &config);
            let request = match into_open_ai(
                request,
                &config.name,
                config.capabilities.parallel_tool_calls,
                config.capabilities.prompt_cache_key,
                config.max_output_tokens,
                chat_completion_max_tokens_parameter(&config),
                reasoning_effort,
                config.capabilities.interleaved_reasoning,
            ) {
                Ok(request) => request,
                Err(error) => return async move { Err(error.into()) }.boxed(),
            };
            let completions = self.stream_chat_completion(&request_limiter, request, cx);
            let executor = cx.background_executor().clone();
            async move {
                let mapper = ChatCompletionEventMapper::new();
                Ok(language_model::stream_in_background(
                    mapper.map_stream(completions.await?).boxed(),
                    executor,
                ))
            }
            .boxed()
        } else {
            disable_response_thinking_for_none_effort(&mut request, &config);
            let request = match into_open_ai_response(
                request,
                &config.name,
                config.capabilities.parallel_tool_calls,
                config.capabilities.prompt_cache_key,
                config.max_output_tokens,
                default_thinking_reasoning_effort(&config),
                supports_none_reasoning_effort(&config),
                &self.id,
            ) {
                Ok(request) => request,
                Err(error) => return async move { Err(error.into()) }.boxed(),
            };
            let completions = self.stream_response(&request_limiter, request, cx);
            let compaction_state_owner = self.id.clone();
            let executor = cx.background_executor().clone();
            async move {
                let mapper = OpenAiResponseEventMapper::new(compaction_state_owner);
                Ok(language_model::stream_in_background(
                    mapper.map_stream(completions.await?).boxed(),
                    executor,
                ))
            }
            .boxed()
        }
    }
}

fn default_thinking_reasoning_effort(model: &AvailableModel) -> Option<open_ai::ReasoningEffort> {
    model
        .reasoning_effort
        .filter(|effort| *effort != open_ai::ReasoningEffort::None)
}

fn supported_thinking_effort_levels(model: &AvailableModel) -> Vec<LanguageModelEffortLevel> {
    let Some(default_effort) = default_thinking_reasoning_effort(model) else {
        return Vec::new();
    };

    open_ai::ReasoningEffort::OPENAI_COMPATIBLE_SELECTABLE
        .into_iter()
        .map(|effort| LanguageModelEffortLevel {
            name: effort.label().into(),
            value: effort.value().into(),
            is_default: effort == default_effort,
        })
        .collect()
}

fn selected_thinking_reasoning_effort(
    request: &LanguageModelRequest,
) -> Option<open_ai::ReasoningEffort> {
    request
        .thinking_effort
        .as_deref()
        .and_then(|effort| effort.parse::<open_ai::ReasoningEffort>().ok())
        .filter(|effort| *effort != open_ai::ReasoningEffort::None)
}

fn chat_completion_max_tokens_parameter(
    model: &AvailableModel,
) -> crate::provider::open_ai::ChatCompletionMaxTokensParameter {
    if model.capabilities.max_tokens_parameter {
        crate::provider::open_ai::ChatCompletionMaxTokensParameter::MaxTokens
    } else {
        crate::provider::open_ai::ChatCompletionMaxTokensParameter::MaxCompletionTokens
    }
}

fn supports_none_reasoning_effort(model: &AvailableModel) -> bool {
    model.reasoning_effort.is_some()
}

fn chat_completion_reasoning_effort(
    request: &LanguageModelRequest,
    model: &AvailableModel,
) -> Option<open_ai::ReasoningEffort> {
    if model.reasoning_effort == Some(open_ai::ReasoningEffort::None) {
        return Some(open_ai::ReasoningEffort::None);
    }

    if request.thinking_allowed {
        selected_thinking_reasoning_effort(request)
            .or_else(|| default_thinking_reasoning_effort(model))
    } else if supports_none_reasoning_effort(model) {
        Some(open_ai::ReasoningEffort::None)
    } else {
        None
    }
}

fn disable_response_thinking_for_none_effort(
    request: &mut LanguageModelRequest,
    model: &AvailableModel,
) {
    if model.reasoning_effort == Some(open_ai::ReasoningEffort::None) {
        request.thinking_allowed = false;
        request.thinking_effort = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    fn available_model(reasoning_effort: Option<open_ai::ReasoningEffort>) -> AvailableModel {
        AvailableModel {
            name: "custom-model".to_string(),
            display_name: None,
            max_tokens: 128_000,
            max_output_tokens: None,
            max_completion_tokens: None,
            reasoning_effort,
            capabilities: ModelCapabilities {
                chat_completions: false,
                ..Default::default()
            },
        }
    }

    #[test]
    fn configured_reasoning_effort_supports_thinking() {
        assert_eq!(
            default_thinking_reasoning_effort(&available_model(Some(
                open_ai::ReasoningEffort::High
            ))),
            Some(open_ai::ReasoningEffort::High)
        );
    }

    #[test]
    fn missing_or_none_reasoning_effort_does_not_support_thinking() {
        assert_eq!(
            default_thinking_reasoning_effort(&available_model(None)),
            None
        );
        assert_eq!(
            default_thinking_reasoning_effort(&available_model(Some(
                open_ai::ReasoningEffort::None
            ))),
            None
        );
    }

    #[test]
    fn supported_thinking_effort_levels_use_configured_effort_as_default() {
        let effort_levels = supported_thinking_effort_levels(&available_model(Some(
            open_ai::ReasoningEffort::High,
        )));
        let values = effort_levels
            .iter()
            .map(|level| level.value.as_ref())
            .collect::<Vec<_>>();

        assert_eq!(values, ["minimal", "low", "medium", "high", "xhigh", "max"]);
        assert_eq!(
            effort_levels
                .iter()
                .find(|level| level.is_default)
                .map(|level| level.value.as_ref()),
            Some("high")
        );
    }

    #[test]
    fn supported_thinking_effort_levels_hide_missing_or_none_effort() {
        assert!(supported_thinking_effort_levels(&available_model(None)).is_empty());
        assert!(
            supported_thinking_effort_levels(&available_model(Some(
                open_ai::ReasoningEffort::None
            )))
            .is_empty()
        );
    }

    #[test]
    fn chat_completion_reasoning_effort_honors_request_and_configured_effort() {
        let model = available_model(Some(open_ai::ReasoningEffort::Medium));
        let mut request = LanguageModelRequest {
            thinking_allowed: true,
            ..Default::default()
        };

        assert_eq!(
            chat_completion_reasoning_effort(&request, &model),
            Some(open_ai::ReasoningEffort::Medium)
        );

        request.thinking_effort = Some("high".to_string());
        assert_eq!(
            chat_completion_reasoning_effort(&request, &model),
            Some(open_ai::ReasoningEffort::High)
        );

        request.thinking_effort = Some("not-supported".to_string());
        assert_eq!(
            chat_completion_reasoning_effort(&request, &model),
            Some(open_ai::ReasoningEffort::Medium)
        );

        request.thinking_allowed = false;
        assert_eq!(
            chat_completion_reasoning_effort(&request, &model),
            Some(open_ai::ReasoningEffort::None)
        );
    }

    #[test]
    fn chat_completion_reasoning_effort_omits_missing_effort() {
        let model = available_model(None);
        let request = LanguageModelRequest {
            thinking_allowed: false,
            ..Default::default()
        };

        assert_eq!(chat_completion_reasoning_effort(&request, &model), None);
    }

    #[test]
    fn chat_completion_reasoning_effort_preserves_explicit_none() {
        let model = available_model(Some(open_ai::ReasoningEffort::None));
        let request = LanguageModelRequest {
            thinking_allowed: true,
            thinking_effort: Some("high".to_string()),
            ..Default::default()
        };

        assert_eq!(
            chat_completion_reasoning_effort(&request, &model),
            Some(open_ai::ReasoningEffort::None)
        );
    }

    #[test]
    fn chat_completion_max_tokens_parameter_defaults_to_max_completion_tokens() {
        let model = available_model(Some(open_ai::ReasoningEffort::Medium));

        assert_eq!(
            chat_completion_max_tokens_parameter(&model),
            crate::provider::open_ai::ChatCompletionMaxTokensParameter::MaxCompletionTokens
        );
    }

    #[test]
    fn chat_completion_max_tokens_parameter_uses_max_tokens_when_configured() {
        let mut model = available_model(Some(open_ai::ReasoningEffort::Medium));
        model.capabilities.max_tokens_parameter = true;

        assert_eq!(
            chat_completion_max_tokens_parameter(&model),
            crate::provider::open_ai::ChatCompletionMaxTokensParameter::MaxTokens
        );
    }

    #[test]
    fn response_request_includes_reasoning_when_effort_is_configured() {
        let model = available_model(Some(open_ai::ReasoningEffort::High));
        let request = LanguageModelRequest {
            thinking_allowed: true,
            ..Default::default()
        };

        let request = into_open_ai_response(
            request,
            &model.name,
            model.capabilities.parallel_tool_calls,
            model.capabilities.prompt_cache_key,
            model.max_output_tokens,
            default_thinking_reasoning_effort(&model),
            supports_none_reasoning_effort(&model),
            &LanguageModelProviderId::new("test-compatible-provider"),
        )
        .unwrap();
        let serialized = serde_json::to_value(request).unwrap();

        assert_eq!(
            serialized["reasoning"],
            json!({ "effort": "high", "summary": "auto" })
        );
        assert_eq!(
            serialized["include"],
            json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn response_request_omits_reasoning_when_effort_is_missing() {
        let model = available_model(None);
        let request = LanguageModelRequest {
            thinking_allowed: true,
            ..Default::default()
        };

        let request = into_open_ai_response(
            request,
            &model.name,
            model.capabilities.parallel_tool_calls,
            model.capabilities.prompt_cache_key,
            model.max_output_tokens,
            default_thinking_reasoning_effort(&model),
            supports_none_reasoning_effort(&model),
            &LanguageModelProviderId::new("test-compatible-provider"),
        )
        .unwrap();
        let serialized = serde_json::to_value(request).unwrap();

        assert_eq!(serialized.get("reasoning"), None);
        assert_eq!(serialized.get("include"), None);
    }

    #[test]
    fn chat_completion_request_includes_selected_reasoning_effort() {
        let mut model = available_model(Some(open_ai::ReasoningEffort::Medium));
        model.capabilities.chat_completions = true;
        let request = LanguageModelRequest {
            thinking_allowed: true,
            thinking_effort: Some("high".to_string()),
            ..Default::default()
        };
        let reasoning_effort = chat_completion_reasoning_effort(&request, &model);

        let request = into_open_ai(
            request,
            &model.name,
            model.capabilities.parallel_tool_calls,
            model.capabilities.prompt_cache_key,
            model.max_output_tokens,
            chat_completion_max_tokens_parameter(&model),
            reasoning_effort,
            model.capabilities.interleaved_reasoning,
        )
        .unwrap();
        let serialized = serde_json::to_value(request).unwrap();

        assert_eq!(serialized["reasoning_effort"], json!("high"));
    }

    #[test]
    fn configured_reasoning_effort_supports_none_reasoning_effort() {
        assert!(supports_none_reasoning_effort(&available_model(Some(
            open_ai::ReasoningEffort::Medium
        ))));
        assert!(supports_none_reasoning_effort(&available_model(Some(
            open_ai::ReasoningEffort::None
        ))));
        assert!(!supports_none_reasoning_effort(&available_model(None)));
    }

    #[test]
    fn response_thinking_effort_preserves_explicit_none() {
        let model = available_model(Some(open_ai::ReasoningEffort::None));
        let mut request = LanguageModelRequest {
            thinking_allowed: true,
            thinking_effort: Some("high".to_string()),
            ..Default::default()
        };

        disable_response_thinking_for_none_effort(&mut request, &model);
        assert!(!request.thinking_allowed);
        assert_eq!(request.thinking_effort, None);
    }
}
