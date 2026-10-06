# Least privilege: SSM core (for Session Manager), CloudWatch agent metrics, and read-only S3 on one prefix.
data "aws_iam_policy_document" "assume_ec2" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "node" {
  name_prefix        = "s2s-node-"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy_attachment" "cw_agent" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

data "aws_iam_policy_document" "weights_read" {
  count = var.weights_bucket == "" ? 0 : 1
  statement {
    actions   = ["s3:ListBucket"]
    resources = ["arn:aws:s3:::${var.weights_bucket}"]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["${var.weights_prefix}*"]
    }
  }
  statement {
    actions   = ["s3:GetObject"]
    resources = ["arn:aws:s3:::${var.weights_bucket}/${var.weights_prefix}*"]
  }
}

resource "aws_iam_role_policy" "weights_read" {
  count  = var.weights_bucket == "" ? 0 : 1
  role   = aws_iam_role.node.id
  policy = data.aws_iam_policy_document.weights_read[0].json
}

# The HF token lives in SSM Parameter Store (SecureString), never in this repo or in user_data.
# Create it once:  aws ssm put-parameter --name /s2s/hf_token --type SecureString --value "$HF_TOKEN"
data "aws_iam_policy_document" "hf_token" {
  statement {
    actions   = ["ssm:GetParameter"]
    resources = ["arn:aws:ssm:${var.region}:*:parameter/s2s/hf_token"]
  }
}

resource "aws_iam_role_policy" "hf_token" {
  role   = aws_iam_role.node.id
  policy = data.aws_iam_policy_document.hf_token.json
}

resource "aws_iam_instance_profile" "node" {
  name_prefix = "s2s-node-"
  role        = aws_iam_role.node.name
}
