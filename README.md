## AWS Redshift: Native and External Tables Support

This version supports querying both native Redshift tables and external tables defined via AWS Glue Data Catalog. External tables are typically created from files uploaded to an S3 bucket.

### Using External Tables

1. **Upload your data files to an S3 bucket.**
2. **Run the AWS Glue Crawler** on the uploaded files to create/update the Data Catalog database and tables.
3. **Set the environment variable `REDSHIFT_AWSCATALOG_DATABASE`** to the name of the database created/populated by the Glue Crawler. This enables the application to query external tables in addition to native Redshift tables.
4. Using the external database is optional. If you do not set `REDSHIFT_AWSCATALOG_DATABASE`, only native Redshift tables will be used.

### Environment Variable Example

```
REDSHIFT_AWSCATALOG_DATABASE=your_glue_database_name
```

### Summary

- Native tables: Standard Redshift tables in your cluster.
- External tables: Tables defined in AWS Glue Data Catalog, typically from S3 files.
- Both types can be queried if `REDSHIFT_AWSCATALOG_DATABASE` is set.

For more details, see the AWS Glue and Redshift documentation.
# Agent-query-data warehouse

This project is a GenAI-powered agent that answers natural language questions by querying a database. It is designed for extensibility and can be adapted to different database backends.

It can work with schemas with many tables and many columns. It works in 3 steps. First it determines the needed tables for the user query. Then it builds the SQL and query database. At the end it formats the output.

## Redshift Data API (Serverless)

This version uses the **AWS Redshift Data API** to connect to Redshift Serverless clusters, including those inside a VPC. The Data API works over HTTP, allowing secure access from anywhere (local machine, cloud, CI/CD, etc.) without requiring a direct network connection or VPN. This is ideal for serverless and cloud-native architectures.

- **Why Data API?**
  - Connects to Redshift Serverless inside a VPC using HTTP.
  - No need for JDBC drivers, SSH tunnels, or direct network access.
  - Works with IAM authentication and AWS credentials.
  - Supports stateless, asynchronous query execution.

## How it Works
- The agent uses LangChain and OpenAI to translate user questions into SQL queries.
- SQL queries are executed against Redshift using the Data API.
- The agent automatically fetches schema information to help the LLM generate accurate queries.
- Conversation history is maintained for context.


## Database Backend Support

This app was originally created to work with AWS Redshift using the Redshift Data API. All core logic and schema extraction was first designed for Redshift.

Later, support for Azure SQL Server was added by creating a separate `db_utils_azure.py` library. The app can now be extended to other databases by implementing a new `db_utils_<your_db>.py` module and updating the import in `Agent.py`.

### How to Switch Database Backends

1. For Redshift, ensure you are importing from `db_utils_redshift` in `Agent.py`:
   ```python
   from db_utils_redshift import ...
   ```

2. For Azure SQL Server, import from `db_utils_azure`:
   ```python
   from db_utils_azure import ...
   ```

3. For other databases, create a new `db_utils_<your_db>.py` with the required functions (`get_tables`, `get_columns`, `query_database`, etc.), and update the import in `Agent.py` accordingly.

### Azure SQL Server Setup

To use Azure SQL Server, you need to install the ODBC driver:
```sh
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew update
ACCEPT_EULA=Y brew install msodbcsql18
```

Update your `.env` file with the correct Azure SQL connection variables.


## Setup
1. Clone the repository.
2. Create a `.env` file with your credentials and model configuration:
   ```ini
   # AWS Redshift
   AWS_REGION=your-region
   REDSHIFT_WORKGROUP_NAME=your-workgroup
   REDSHIFT_DATABASE=your-database
   REDSHIFT_SCHEMA=your-schema
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key

   # OpenAI
   OPENAI_API_KEY=your-openai-key

   # LLM Provider and Model Selection
   # Choose which LLM provider and model to use:
   # Supported providers: openai, bedrock
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-4o

   # For AWS Bedrock (if LLM_PROVIDER=bedrock)
   BEDROCK_PROVIDER=anthropic         # e.g., anthropic, meta, amazon, etc.
   BEDROCK_REGION=us-east-1
   BEDROCK_INFERENCE_PROFILE_ID=your-inference-profile-id-or-arn
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the agent:
   ```sh
   python src/Agent.py
   ```


## Model and Provider Configuration

You can select which LLM provider and model to use by setting the following environment variables in your `.env` file:

- `LLM_PROVIDER`: Set to `openai` to use OpenAI models, or `bedrock` to use AWS Bedrock models.
- `LLM_MODEL`: The model name for OpenAI (e.g., `gpt-4o`, `gpt-4.1`).
- `BEDROCK_PROVIDER`: The model provider for Bedrock (e.g., `anthropic`, `meta`, `amazon`).
- `BEDROCK_REGION`: AWS region for Bedrock (default: `us-east-1`).
- `BEDROCK_INFERENCE_PROFILE_ID`: (Optional) The inference profile ID or ARN for Bedrock models that require it.

Example for OpenAI:
```ini
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
```

Example for AWS Bedrock (with inference profile):
```ini
LLM_PROVIDER=bedrock
BEDROCK_PROVIDER=anthropic
BEDROCK_REGION=us-east-1
BEDROCK_INFERENCE_PROFILE_ID=arn:aws:bedrock:us-east-1:840245022720:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0
```

## Comments
- Schema should use names that are clear. If further data is needed use database COMMENT on schema, table and column level to describe them. If you wish to remove tables from queries, add comment with 'hidden' text.

   COMMENT ON SCHEMA my_schema IS 'This is a schema comment';
   COMMENT ON TABLE my_schema.my_table IS 'This is a table comment';
   COMMENT ON COLUMN my_schema.my_table.my_column IS 'This is a column comment';

## Notes
- The agent is stateless and works with Redshift Serverless.
- For other databases, replace the Redshift-specific code with your preferred connection method.

## License
MIT
