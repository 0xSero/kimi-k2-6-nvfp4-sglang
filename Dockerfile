ARG SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.12.post1
FROM ${SGLANG_BASE_IMAGE}

COPY docker/entrypoint.sh /usr/local/bin/kimi-k2-6-sglang
RUN chmod +x /usr/local/bin/kimi-k2-6-sglang

ENTRYPOINT ["/usr/local/bin/kimi-k2-6-sglang"]
