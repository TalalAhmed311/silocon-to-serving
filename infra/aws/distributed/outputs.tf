output "checkpoint_bucket" { value = aws_s3_bucket.ckpt.bucket }
output "train_role_arn" { value = aws_iam_role.train.arn }
