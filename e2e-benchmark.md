## Running containers

### Running AMD vLLM container.
Follow https://github.com/powderluv/vllm-docs/tree/main
Do not export PYTORCH_TUNABLEOP_* env vars as the benchmark uses variable-sized inputs.

### Running NVIDIA NIM container.
Follow NIM docs: https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html 

For example for llama 8b: https://build.nvidia.com/meta/llama3-8b?snippet_tab=Docker 
to run on 1 gpu:
```
docker run -it --rm --ulimit memlock=-1  --privileged   \
  --gpus 1 --shm-size=16GB  -e NGC_API_KEY="$NGC_API_KEY"   \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" -u $(id -u)     \
  -p 8000:8000     \
  nvcr.io/nim/meta/llama3-8b-instruct:latest
```


## Benchmarking

### Fireworks benchmarking tool
https://github.com/fw-ai/benchmark
```
locust --provider vllm -H http://localhost:8000 \
  --summary-file /tmp/load_test.csv \
  --max-tokens-distribution uniform --max-tokens-range 0.1 \
  -r 1 -p 2048 -o 128 -u 32 -t 1m
```

### llmperf benchmarking tool
https://github.com/ray-project/llmperf
For AMD vLLM container replace $MODEL with model path on disk.
For NVIDIA NIM container replace $MODEL with its name, e.g. meta/llama3-8b-instruct.
Tip: models can be always queries by calling http://localhost:8000/v1/models API.
```
OPENAI_API_KEY=NONE OPENAI_API_BASE=http://localhost:8000/v1 python token_benchmark_ray.py \
  --model "$MODEL" \
  --mean-input-tokens 2048 \
  --stddev-input-tokens 200 \
  --mean-output-tokens 128 \
  --stddev-output-tokens 10 \
  --max-num-completed-requests 2000 \
  --timeout 60 \
  --num-concurrent-requests 32 \
  --llm-api openai
```
