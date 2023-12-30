import mlflow
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import joblib

#create dataset
training_data=[(2,10),(2,6),(11,11),(6,9),(6,4),(1,2),(5,10),(4,9),(10,12),(7,5),(9,11),(4,6),(3,10),(3,8),(6,11)]
#create a new experiment
experiment_name = 'ClusteringWithMlflow'
try:
    exp_id = mlflow.create_experiment(name=experiment_name)
except Exception as e:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
#run the code for different number of clusters
range_of_k = range(2,4) 
best_silhouette_score = 0
best_model = None
for k in range_of_k :
    with mlflow.start_run(experiment_id=exp_id):
        untrained_model = KMeans(n_clusters=k)
        trained_model=untrained_model.fit(training_data)
        cluster_labels = trained_model.labels_
        score=silhouette_score(training_data, cluster_labels)
        #save parameter
        mlflow.log_param('value_of_k', k)
        #save metric
        mlflow.log_metric('silhoutte_score', score)
        #save model
        mlflow.sklearn.log_model(trained_model, "Clustering_Model")
        #end current run
        if score > best_silhouette_score:
            print("score:",score)
            best_model = trained_model
            best_silhouette_score = score
            model_path = f"models/kmeans_model.pkl"
            joblib.dump(trained_model, model_path)
        mlflow.end_run()
if best_model is not None:
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="sklearn-model",
        # signature=signature,
        registered_model_name="Best_KMeans_Model",
    )
    # mlflow.register_model(f"runs:/{mlflow.active_run().info.run_id}/Clustering_Model", "Best_KMeans_Model")

# client = mlflow.tracking.MlflowClient()
# runs = client.search_runs(exp_id, "", order_by=["metrics.rmse DESC"], max_results=1)
# best_run = runs[0]
# print(best_run)