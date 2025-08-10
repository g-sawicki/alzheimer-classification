output "eks_cluster_name" {
  value = module.eks.cluster_name
}

output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

# output "mlflow_s3_bucket" {
#   value = aws_s3_bucket.mlflow_artifacts.bucket
# }
#
# output "mlflow_db_endpoint" {
#   value = aws_db_instance.mlflow.address
# }
