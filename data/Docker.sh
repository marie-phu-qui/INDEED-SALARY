# Build the current dockerfile
docker build -t dev/marie:0.0.0 .

# Run the dockerfile
docker run dev/marie:0.0.0

# Stop the container
# docker stop dev/marie:0.0.0