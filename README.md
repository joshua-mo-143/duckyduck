## duckyduck
Duckyduck is a WIP knowledge base for exploring codebases by utilising semantic code search, as well as asking it questions that can leverage the power of the knowledge base through self-querying retrieval and other modern RAG techniques.

## Usage
Note that this repo has technically not reached MVP yet, so you will probably encounter bugs. However if you're not afraid, read on!

You will need the following:
- A local Qdrant instance where the gRPC port maps to localhost:6334 (this can be done easily).
- A Huggingface API token (set with `HF_TOKEN`).

Once you're done, simply use `cargo run --bin cli embed` to embed the current repo into your Qdrant instance.

After that, try using `cargo run --bin cli search <prompt>` or `cargo run --bin server` to load up the web server at `localhost:8000`, which contains a prompt input you can try out to fetch stuff from the codebase.

## Features
- [x] Basic code search 
- [ ] Prompting model usage
- [ ] Conversation history
- [ ] Uploading Github repos to Qdrant
- [ ] Uploading markdown documentation into Qdrant
- [ ] Parsing other languages than Rust
