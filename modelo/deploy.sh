clear

kubectl delete job model-trainer-data-base
kubectl apply -f k-trainer-deployment.yaml 

kubectl apply -f k-front-deployment.yaml