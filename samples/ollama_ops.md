# Running Ollama Locally

## Keeping models in memory

By default Ollama unloads a model five minutes after its last request, so the next call pays the
full load time again. Set OLLAMA_KEEP_ALIVE to a longer duration such as 30m to keep it resident.

## Limiting memory use

OLLAMA_MAX_LOADED_MODELS controls how many models stay in memory at once. With a separate embedding
model, set it to 2 so embedding calls do not evict the chat model.

## Troubleshooting

Error E-4021 means the model file is missing from the local library. Run `ollama pull` with the
model name and retry. Error E-4107 means the context window was exceeded; reduce the prompt or
raise num_ctx.

If generation is slow on a laptop, check whether the model is running on the CPU or the GPU with
`ollama ps`.
