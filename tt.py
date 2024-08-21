from huggingface_hub import snapshot_download
import os

snapshot_download(repo_id="facebook/bart-large", cache_dir='/Users/sesh/Documents/neyro/model/hug')