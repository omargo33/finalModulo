docker build -t omargo33/trainer-final:1.0.9 -f Dockerfile.trainer .
docker push omargo33/trainer-final:1.0.9

docker build -t omargo33/front-final:1.0.9 -f Dockerfile.front .
docker push omargo33/front-final:1.0.9
