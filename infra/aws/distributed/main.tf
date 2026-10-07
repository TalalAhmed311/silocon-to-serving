# P4.3 add-on to the P3.2 EKS cluster: a private, versioned, encrypted S3 bucket for checkpoints and Ray run state,
# and an EKS Pod Identity role that lets ONLY the s2s-train/s2s-train ServiceAccount read and write it.
# The GPU capacity itself comes from infra/aws/eks: set gpu_instance_types = ["g6.12xlarge"] (4 GPUs per node) there.

data "aws_caller_identity" "me" {}

resource "aws_s3_bucket" "ckpt" {
  bucket        = "s2s-ckpt-${var.owner}-${data.aws_caller_identity.me.account_id}"
  force_destroy = true # lab bucket: `make down` must be able to delete it with its checkpoints
}

resource "aws_s3_bucket_public_access_block" "ckpt" {
  bucket                  = aws_s3_bucket.ckpt.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "ckpt" {
  bucket = aws_s3_bucket.ckpt.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "ckpt" {
  bucket = aws_s3_bucket.ckpt.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "ckpt" {
  bucket = aws_s3_bucket.ckpt.id
  rule {
    id     = "expire-old-checkpoints"
    status = "Enabled"
    filter {}
    expiration { days = var.checkpoint_retention_days }
    noncurrent_version_expiration { noncurrent_days = 1 }
    abort_incomplete_multipart_upload { days_after_initiation = 1 }
  }
}

data "aws_iam_policy_document" "trust" {
  statement {
    actions = ["sts:AssumeRole", "sts:TagSession"]
    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "train" {
  name               = "s2s-train-${var.owner}"
  assume_role_policy = data.aws_iam_policy_document.trust.json
}

data "aws_iam_policy_document" "ckpt" {
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.ckpt.arn]
  }
  statement {
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = ["${aws_s3_bucket.ckpt.arn}/*"]
  }
}

resource "aws_iam_role_policy" "ckpt" {
  role   = aws_iam_role.train.id
  policy = data.aws_iam_policy_document.ckpt.json
}

resource "aws_eks_pod_identity_association" "train" {
  cluster_name    = var.cluster_name
  namespace       = "s2s-train"
  service_account = "s2s-train"
  role_arn        = aws_iam_role.train.arn
}
