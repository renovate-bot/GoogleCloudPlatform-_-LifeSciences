ARG OF3_VERSION="0.3.1"
FROM openfoldconsortium/openfold3:${OF3_VERSION}

# Install HMMER suite (jackhmmer, nhmmer) with OpenBLAS dependency for MSA search.
# The base OF3 image includes a jackhmmer binary but is missing libopenblas.so.0,
# causing exit code 127 at runtime. Installing hmmer via apt pulls in all required libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        hmmer \
    && rm -rf /var/lib/apt/lists/*

# Do NOT bake in weights — they come from NFS at runtime
# Do NOT bake in databases — they come from NFS
