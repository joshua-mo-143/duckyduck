docker-qdrant:
	docker run -d -t -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

up:
	cargo run --bin server

embed:
	cargo run --bin cli embed
