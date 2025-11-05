# src/data_acquisition/web/web_constants.py

# Seed URLs for crawling
SEED_URLS = [
    "https://igotanoffer.com/blogs/tech/system-design-interviews",
    "https://www.tryexponent.com/blog/system-design-interview-guide",
    "https://www.designgurus.io/answers/detail/practical-system-design-case-studies-with-guided-solutions",
    "https://github.com/ashishps1/awesome-system-design-resources",
    "https://dev.to/kumarkalyan/10-engineering-blogs-to-become-a-system-design-hero-for-free-20ee",
    "https://www.geeksforgeeks.org/category/system-design/",
    "https://www.geeksforgeeks.org/system-design/most-commonly-asked-system-design-interview-problems-questions/",
    "https://www.adaface.com/blog/system-design-interview-questions/",
    "https://github.com/sid24rane/System-Design-Interview-Questions",
    "https://www.hellointerview.com/learn/system-design/in-a-hurry/introduction",
    "https://highscalability.com/untitled-2/",
    "https://github.com/karanpratapsingh/system-design",
    "https://github.com/donnemartin/system-design-primer"

]

# Keywords to identify relevant URLs
KEYWORDS = [
    "system-design", "architecture", "scalable", "microservices",
    "interview", "backend", "load-balancing", "database", "cache",
    "geeksforgeeks.org/system-design"
]

# Output file names
URL_PROGRESS_FILE = "url_discovery_progress.json"
URL_DISCOVERY_OUTPUT = "web_sources.csv"
WEB_SCRAPPER_HISTORY = "web_crawl_history.json"
WEB_SCRAPER_OUTPUT = "web_scraper_output.csv"

# Limits, timeout, and headers
MAX_LINKS_PER_SOURCE = 100
REQUEST_TIMEOUT = 10
USER_AGENT = "Mozilla/5.0 (compatible; TopicModelBot/1.2)"

# Domains that require JS rendering
# JS_REQUIRED_DOMAINS = []

VIDEO_DOMAINS = [
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "dailymotion.com",
    "twitch.tv"
]

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
    "scalability bottleneck","horizontal scaling","vertical scaling", "performance tuning","load distribution", "throughput optimization", "consistency model",

    # Data storage & databases
    "database", "sql", "nosql", "postgresql", "mysql", "mongodb",
    "redis", "cassandra", "dynamodb", "elastic search", "elasticsearch",
    "data replication", "read replicas", "write replicas", "leader election",
    "consensus", "raft", "paxos", "zookeeper", "schema design",
    "data modeling", "index optimization", "query optimization",

    # Messaging / communication
    "pub sub","pubsub","pub/sub", "message queue", "kafka", "rabbitmq", "event-driven",
    "stream processing", "batch processing", "data pipeline",
    "real-time analytics", "message broker", "producer consumer",

    # APIs and services
    "api", "rest api", "graphql", "grpc", "thrift", "microservices",
    "monolith","monolithic", "service discovery", "api gateway", "service mesh",
    "container","containerization","docker", "kubernetes", "orchestration", "serverless"
    "lambda", "autoscaling", "deployment", "cicd","ci/cd","load testing","endpoint", "latency budget",
    "service dependency", "api request", "api response",

    # Networking / Infra
    "cdn", "dns", "edge network", "proxy", "reverse proxy",
    "http", "websocket", "tcp", "udp", "ip routing", "firewall",
    "availability zone", "region", "multi-region", "data center", "bandwidth", "concurrency",

    # Monitoring / reliability
    "observability", "logging", "metrics", "distributed tracing",
    "monitoring", "alerting", "dashboard", "chaos engineering",
    "sla", "retries", "timeout", "circuit breaker",

    # architecture/system design terms
    "architecture", "scalability", "latency", "throughput",
    "cache", "replication", "sharding", "load balancer",
    "consistency", "availability", "microservice", "api",
    "cdn", "queue", "database", "redis", "message broker",
    "fault tolerance", "system design", "rate limiter",
    "partition", "storage", "distributed", "aws", "gcp",
    "azure", "kafka", "rabbitmq", "elastic", "search", "index",
    "replica", "latency", "bandwidth", "concurrency",
    "query", "indexing", "rps", "qps", "sla", "eventual consistency",

    # Interview / contextual terms
    "interview", "candidate", "interviewer", "approach", "framework",
    "tradeoff", "bottleneck", "scalability challenge", "design question",
    "tips", "strategy", "thinking", "estimate", "requirements gathering",
    "clarify", "constraints", "assumptions", "data flow",
    "component diagram", "high level design", "low level design",
    "estimation", "storage estimate", "latency requirement",
    "availability requirement", "design approach", "high level overview",
    "follow up question", "trade off", "capacity estimation",
    "bottleneck analysis", "scaling strategy", "design considerations",
    "clarifying question",

    # Design theory / patterns
    "cap theorem", "acid", "base", "idempotence",
    "lock", "mutex","observer pattern", "throttling"

    # Case studies / real systems
    "design instagram", "design uber", "design whatsapp", "design netflix",
    "design youtube", "design dropbox", "design twitter", "design slack",
    "design zoom", "design reddit", "design booking", "design linkedin",
    "design messenger", "design tinder", "design spotify",
    "design pinterest", "design tiktok", "design amazon",

    # ML / Recommendation systems
    "machine learning", "ml", "artificial intelligence", "recommendation system",
    "recommender system", "collaborative filtering", "ranking",
    "personalization", "model training", "feature engineering",
    "content-based filtering", "deep learning", "vector similarity",
    "embedding", "semantic search", "retrieval", "clustering",
    "inference", "serving layer"

]

UNWANTED_KEYWORDS = [
    "refund", "credits", "trial", "coaching", "offer", "risk-free",
    "subscribe", "advert", "sponsor", "free", "buy", "purchase",
    "register", "sign up", "promo", "discount", "testimonial",
    "learn more", "contact us", "about us", "cookie policy",
    "newsletter", "privacy policy", "terms of service", "check out", "course",
    "your dashboard", "pricing", "sign in", "sign up" , "community" ,"coaches"
]

HARD_BLOCK_KEYWORDS = [
    "refund", "credits", "trial", "offer", "risk-free",
    "subscribe", "advert", "sponsor", "free", "buy", "purchase",
    "register", "promo", "discount", "testimonial","learn more",
    "contact us", "about us","cookie policy",
    "privacy policy", "terms of service",
    "upi","deal","premium","all rights reserved","cookie consent","optick labs"
]

SOFT_BLOCK_KEYWORDS = [
    "course", "coaching", "community", "pricing", "sign up", "sign in", "dashboard",
    "privacy", "terms", "cookies", "policy", "contact","faq", "sitemap", "support",
    "check out","payment",
]


UI_NOISE_WORDS = ["home", "dashboard", "menu", "login", "logout", "signup", "share", "next", "previous"]

MIN_PARAGRAPH_LENGTH = 120
MIN_PARAGRAPH_WORDS = 15

FAILED_URLS_AFTER_MAX_RETRIES = [
    'https://blog.wahab2.com/api-architecture-best-practices-for-designing-rest-apis-bf907025f5f',
    'https://onesearch.blog'
]