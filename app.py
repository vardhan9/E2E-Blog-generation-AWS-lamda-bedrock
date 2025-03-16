import json
import boto3
import botocore.config
from datetime import datetime

def blog_generate_using_bedrock(blogtopic: str) -> str:
    prompt = f"""<s>[INST]Human: Write 200 words blog on {blogtopic}
    Assistant:[/INST]"""

    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            config=botocore.config.Config(read_timeout=300, retries={"max_attempts": 3})
        )

        response = bedrock.invoke_model(
            body=json.dumps(body), modelId="meta.llama3-8b-instruct-v1:0"
        )

        response_content = response["body"].read()
        response_data = json.loads(response_content)

        return response_data.get("generation", "No content generated")
    except Exception as e:
        print(f"Error: {e}")
        return "Error in generating blog"

def save_blog_to_s3(s3_bucket: str, s3_key: str, blog_content: str):
    s3 = boto3.client("s3")
    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=blog_content.encode("utf-8"))
        return True
    except Exception as e:
        print(f"Error saving to S3: {e}")
        return False

def lambda_handler(event, context):
    try:
        # Check if body exists (for API Gateway requests)
        if "body" in event and isinstance(event["body"], str):
            event_data = json.loads(event["body"])  # Convert JSON string to dict
        else:
            event_data = event  # Direct invocation case

        if "blog_topic" not in event_data:
            raise KeyError("Missing 'blog_topic' in request payload")

        blogtopic = event_data["blog_topic"]
        generate_blog = blog_generate_using_bedrock(blogtopic=blogtopic)

        if generate_blog:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            s3_key = f"blog_{current_time}.txt"
            s3_bucket = "awsbedrockbloggenai-marthala"
            save_blog_to_s3(s3_bucket, s3_key, generate_blog)
        else:
            return {"statusCode": 500, "body": "Error in generating blog"}

        return {"statusCode": 200, "body": "Blog generated successfully"}
    except KeyError as e:
        print(f"Missing Key: {e}")
        return {"statusCode": 400, "body": f"Bad Request - {e}"}
    except Exception as e:
        print(f"Error in lambda_handler: {e}")
        return {"statusCode": 500, "body": "Internal Server Error"}
