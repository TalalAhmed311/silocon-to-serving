variable "owner" { type = string }
variable "primary_region" {
  type    = string
  default = "us-east-1"
}
variable "secondary_region" {
  type    = string
  default = "us-west-2"
}
variable "hosted_zone_id" {
  type        = string
  description = "Route 53 hosted zone you control (public zone for internet clients)."
}
variable "record_name" {
  type        = string
  description = "e.g. llm.example.com — the name clients use."
}
variable "primary_endpoint" {
  type        = string
  description = "DNS name of the primary region's gateway load balancer (authenticated gateway; SG limited to client CIDRs)."
}
variable "secondary_endpoint" {
  type        = string
  description = "DNS name of the secondary region's gateway load balancer."
}
variable "ttl" {
  type    = number
  default = 30 # low TTL = faster failover for well-behaved resolvers; clients that cache longer won't move (C1 §2)
}
variable "alarm_periods" {
  type    = number
  default = 3 # consecutive 1-minute periods with RegionUp < 1 before the region is declared down
}
