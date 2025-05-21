# Steps followed to run DeepSeek locally

The instructions about how to run DeepSeek using Sglang locally are as follows:
https://github.com/sgl-project/sglang/tree/main/benchmark/deepseek_v3

The model can be found on HF https://huggingface.co/deepseek-ai/DeepSeek-V3

1. I think it is necessary to run it in a docker container because we will probably will need to use nginx as a load balancer in order to have only one port exposed to the outside world.
2. In MN there is no Docker but Singularity but it has [support for Docker images](https://docs.sylabs.io/guides/2.6/user-guide/singularity_and_docker.html). So, we can use the Docker image provided by the authors of DeepSeek.

```bash
singularity pull docker://lmsysorg/sglang:latest
```

From [Salamandra pre-training model](https://github.com/langtech-bsc/salamandra/tree/main/singularity_images)

Important: If you run out of memory while building the image (disk quota exceeded error), try redirecting the cache to a different directory. By default, it points to your home directory ($HOME/.singularity/cache), which tends to be rather small in HPC environments. Simply set the environment variables SINGULARITY_TMPDIR and SINGULARITY_CACHEDIR as follows in order to overcome this issue:

SINGULARITY_TMPDIR=<XXX> SINGULARITY_CACHEDIR=<XXX> singularity pull docker://$CONTAINER_TAG

