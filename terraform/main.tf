###################################
# General
###################################
terraform {
  required_version = ">= 1.12"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.8"
    }
  }
}

provider "aws" {
  region = "eu-north-1"
}

locals {
  tags = {
    Project     = "alz-eks"
    Environment = "development"
    Terraform   = "true"
  }
}

###################################
# VPC + EKS
###################################
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 6.0.0"

  name = "alz-eks-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["eu-north-1a", "eu-north-1b"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = true
  enable_dns_hostnames = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = local.tags
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 21.0"

  name               = "alz-eks-cluster"
  kubernetes_version = "1.33"

  endpoint_private_access                  = true
  endpoint_public_access                   = true
  enable_cluster_creator_admin_permissions = true

  addons = {
    kube-proxy = {
      most_recent       = true
      resolve_conflicts = "OVERWRITE"
    }
    coredns = {
      most_recent       = true
      resolve_conflicts = "OVERWRITE"
    }
    vpc-cni = {
      most_recent       = true
      before_compute    = true
      resolve_conflicts = "OVERWRITE"
    }
    eks-pod-identity-agent = {
      most_recent       = true
      before_compute    = true
      resolve_conflicts = "OVERWRITE"
    }
  }

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    default = {
      instance_types = ["t3.micro"]
      desired_size   = 1
      min_size       = 1
      max_size       = 2
    }
  }

  tags = local.tags
}

# ###################################
# # MLflow storage
# ###################################
# resource "aws_s3_bucket" "mlflow_artifacts" {
#   bucket = "${var.project_name}-mlflow-artifacts"
# }
#
# resource "aws_s3_bucket_versioning" "mlflow" {
#   bucket = aws_s3_bucket.mlflow_artifacts.id
#   versioning_configuration {
#     status = "Disabled"
#   }
# }
#
# resource "aws_db_subnet_group" "mlflow" {
#   name       = "mlflow-db-subnet-group"
#   subnet_ids = module.vpc.private_subnets
# }
#
# resource "aws_db_instance" "mlflow" {
#   identifier           = "${var.project_name}-mlflow-db"
#   engine               = "postgres"
#   instance_class       = "db.t3.micro"
#   allocated_storage    = 10
#   db_subnet_group_name = aws_db_subnet_group.mlflow.name
#   username             = var.db_username
#   password             = var.db_password
#   skip_final_snapshot  = true
#   publicly_accessible  = false
# }
#
# ###################################
# # IAM Role for MLflow Pod (IRSA)
# ###################################
# data "aws_iam_policy_document" "mlflow_s3_access" {
#   statement {
#     actions = ["s3:PutObject", "s3:GetObject", "s3:ListBucket"]
#     resources = [
#       aws_s3_bucket.mlflow_artifacts.arn,
#       "${aws_s3_bucket.mlflow_artifacts.arn}/*"
#     ]
#   }
# }
#
# resource "aws_iam_policy" "mlflow_s3_policy" {
#   name   = "${var.project_name}-mlflow-s3-policy"
#   policy = data.aws_iam_policy_document.mlflow_s3_access.json
# }
#
# module "mlflow_irsa" {
#   source    = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
#   role_name = "${var.project_name}-mlflow-s3-role"
#
#   role_policy_arns = { policy = aws_iam_policy.mlflow_s3_policy.arn }
#   oidc_providers = {
#     main = {
#       provider_arn               = module.eks.oidc_provider_arn
#       namespace_service_accounts = ["default:my-app"]
#     }
#   }
# }
