cd modelo
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -U pipreqs
pipreqs . --force --ignore .venv,.git
pip install -r requirements.txt 


docker login

docker build -t omargo33/trainer-final:1.0.1 -f Dockerfile.trainer .
docker run --rm -v "$(pwd)/modelos_cnn:/app/modelos_cnn" omargo33/trainer-final:1.0.1
docker push omargo33/trainer-final:1.0.1

docker build -t omargo33/front-final:1.0.1 -f Dockerfile.front .
docker run --rm -p 8501:8501 -v "$(pwd)/modelos_cnn:/app/modelos_cnn" omargo33/front-final:1.0.1
docker push omargo33/front-final:1.0.2

minikube start

kubectl delete all --all
kubectl delete pvc --all

kubectl apply -f k-trainer-deployment.yaml 

watch -n 5 kubectl get jobs
kubectl logs -f model-trainer-data-base

kubectl apply -f k-front-deployment.yaml 

kubectl get jobs
kubectl logs -f job/model-frontend-job-???? consultar de job

kubectl get deployments
kubectl get pods
kubectl get services

minikube ip

docker build -t omargo33/trainer-final:1.0.1 -f Dockerfile.trainer .
docker push omargo33/trainer-final:1.0.1

docker build -t omargo33/front-final:1.0.3 -f Dockerfile.front .
docker push omargo33/front-final:1.0.3
