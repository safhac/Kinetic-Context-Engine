# Main Terraform configuration for KCE Infrastructure
provider "aws" {
  region = "us-east-1"
}

# Resource Group for Governance & Ingestion [cite: 847]
resource "aws_resourcegroups_group" "kce_ingestion_rg" {
  name = "kce-ingestion-rg"
  resource_query {
    query = <<JSON
{
  "ResourceTypeFilters": ["AWS::AllSupported"],
  "TagFilters": [
    {
      "Key": "Project",
      "Values": ["KCE"]
    }
  ]
}
JSON
  }
}