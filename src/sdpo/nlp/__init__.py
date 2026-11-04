import os

# shortcut: Hugging Face's Xet download backend stalled mid-file on this machine; plain HTTP
# works. Remove once huggingface_hub's Xet client is reliable here.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
