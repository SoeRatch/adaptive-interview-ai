# src/data_acquisition/web/web_constants.py

# Seed URLs for crawling
SEED_URLS = [
]

# Keywords to identify relevant URLs
KEYWORDS = [
    "system-design", "architecture", "scalable", "microservices",
    "interview", "backend", "load-balancing", "database", "cache",
    "geeksforgeeks.org/system-design"
]

# Output file names
URL_DISCOVERY_OUTPUT = "system_design_sources.csv"
WEB_SCRAPER_OUTPUT = "system_design_corpus.csv"

# Limits, timeout, and headers
MAX_LINKS_PER_SOURCE = 100
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (compatible; TopicModelBot/1.0)"

# Domains that require JS rendering
JS_REQUIRED_DOMAINS = []


# -----------------------------
# Content Filtering Constants
# -----------------------------

RELEVANT_KEYWORDS = [
    # Core design principles
    "system design", "software architecture", "scalability", "availability",
    "reliability", "performance", "latency", "throughput", "load balancer",
    "cache", "caching", "replication", "sharding", "partitioning",
    "indexing", "consistency", "eventual consistency", "strong consistency",
    "fault tolerance", "redundancy", "failover", "high availability",
    "load shedding", "rate limiting", "backpressure","design patterns","distributed","distributed systems",

    # Data storage & databases
    "database", "sql", "nosql", "postgresql", "mysql", "mongodb",
    "redis", "cassandra", "dynamodb", "elastic search", "elasticsearch",
    "data replication", "read replicas", "write replicas", "leader election",
    "consensus", "raft", "paxos", "zookeeper",

    # Messaging / communication
    "pub/sub", "message queue", "kafka", "rabbitmq", "event-driven",
    "stream processing", "batch processing", "data pipeline",
    "real-time analytics",

    # APIs and services
    "api", "rest api", "graphql", "grpc", "thrift", "microservices",
    "monolith","monolithic", "service discovery", "api gateway", "service mesh",
    "container","containerization","docker", "kubernetes", "orchestration", "serverless"
    "lambda", "autoscaling", "deployment", "cicd", "load testing",

    # Networking / Infra
    "cdn", "dns", "edge network", "proxy", "reverse proxy",
    "http", "websocket", "tcp", "udp", "ip routing", "firewall",
    "availability zone", "region", "multi-region", "data center",

    # Design theory / patterns
    "cap theorem", "acid", "base", "idempotence", "concurrency",
    "lock", "mutex", "queue", "producer consumer", "observer pattern",
    "rate limiter", "throttling", "retries", "timeout", "circuit breaker",

    # Monitoring / reliability
    "observability", "logging", "metrics", "distributed tracing",
    "monitoring", "alerting", "dashboard", "chaos engineering",

    # Case studies / real systems
    "design instagram", "design uber", "design whatsapp", "design netflix",
    "design youtube", "design dropbox", "design twitter", "design slack",
    "design zoom", "design reddit", "design booking", "design linkedin",
    "design messenger", "design tinder", "design spotify"
]

UNWANTED_KEYWORDS = [
    "refund", "credits", "trial", "coaching", "offer", "risk-free",
    "subscribe", "advert", "sponsor", "free", "buy", "purchase",
    "register", "sign up", "promo", "discount", "testimonial",
    "learn more", "contact us", "about us", "cookie policy",
    "newsletter", "privacy policy", "terms of service"
]

MIN_PARAGRAPH_LENGTH = 40