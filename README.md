# Agent-query-database

This project is a GenAI-powered agent that answers natural language questions by querying a database. It is designed for extensibility and can be adapted to different database backends.

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

## Adapting to Other Databases
You can modify the agent to use JDBC or other database drivers (e.g., PostgreSQL, MySQL, SQLite) by:
- Replacing the Redshift Data API logic with a JDBC or Python DB-API connection.
- Adjusting the schema introspection function to match your database.
- Updating environment variables and connection details.

## Setup
1. Clone the repository.
2. Create a `.env` file with your AWS and Redshift credentials:
   ```ini
   AWS_REGION=your-region
   REDSHIFT_WORKGROUP_NAME=your-workgroup
   REDSHIFT_DATABASE=your-database
   REDSHIFT_SCHEMA=your-schema
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   OPENAI_API_KEY=your-openai-key
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the agent:
   ```sh
   python src/sql_agent_rda.py
   ```

## Notes
- The agent is stateless and works with Redshift Serverless.
- For other databases, replace the Redshift-specific code with your preferred connection method.

## License
MIT
