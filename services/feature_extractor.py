# --- Standard library imports ---
import math                                   # log2 for entropy
import re                                     # Regular expressions
import ipaddress                              # Check whether a host is an IP address
from collections import Counter               # Count character frequencies
from urllib.parse import urlparse, parse_qs   # Split URLs / parse query strings


# Ordered list of every feature name. The order matters: the training script
# builds its feature table using this exact order, so it must stay in sync
# with the dictionary returned by calculate_features().
FEATURE_NAMES = [
    "url_length",                      # total characters in the URL
    "has_ip_address",                  # 1 if host is an IP address
    "dot_count",                       # number of "." in the URL
    "url_entropy",                     # randomness of the full URL
    "token_count",                     # number of word/number tokens
    "query_param_count",               # number of ?key=value parameters
    "tld_length",                      # length of the top-level domain
    "path_length",                     # length of the path part
    "has_hyphen_in_domain",            # 1 if host contains "-"
    "number_of_digits",                # total digits in the URL
    "tld_popularity",                  # 1 if TLD is a common one
    "suspicious_file_extension",       # 1 if path ends with .exe, .zip, etc.
    "domain_name_length",              # length of the whole hostname
    "percentage_numeric_chars",        # % of URL characters that are digits
    "hyphen_count",                    # number of "-" in the URL
    "special_character_count",         # non-alphanumeric characters in URL
    "path_segment_count",              # number of "/" separated path parts
    "suspicious_keyword_count",        # phishing-style words (login, verify...)
    "domain_entropy",                  # randomness of the hostname
    "path_entropy",                    # randomness of the path
    "domain_digit_count",              # digits in the hostname
    "domain_digit_ratio",              # digits / hostname length
    "domain_hyphen_count",             # hyphens in the hostname
    "domain_hyphen_ratio",             # hyphens / hostname length
    "max_domain_label_length",         # longest dot-separated hostname part
    "max_token_length",                # longest token in the URL
    "query_length",                    # length of the query string
    "fragment_length",                 # length of the #fragment
    "has_port",                        # 1 if an explicit port is present
    "has_at_symbol",                   # 1 if "@" appears in the URL
    "percent_encoded_count",           # count of %XX encoded characters
    "double_slash_path",               # 1 if "//" appears inside the path
    "repeated_separator_count",        # runs like "--", "__", "..", "//"
    "consecutive_digit_count",         # longest run of digits
    "consecutive_alpha_count",         # longest run of letters
    "punycode_flag",                   # 1 if hostname contains "xn--"
    "path_digit_count",                # digits in the path
    "path_special_character_count",    # special characters in the path
    "query_digit_count",               # digits in the query string
    "query_special_character_count",   # special characters in the query string
]


# Common/popular top-level domains. Phishing sites often use less common ones.
COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int",
    "io", "co", "ai", "app", "dev", "me", "info", "biz",
    "xyz", "online", "site", "tech", "store", "cloud",
    "in", "uk", "us", "ca", "au", "de", "fr", "jp",
    "cn", "bd", "ru", "ch", "nl", "it", "es", "br", "eu"
}


# File endings that often indicate a malicious download
SUSPICIOUS_EXTENSIONS = {
    ".exe", ".scr", ".zip", ".rar", ".7z",
    ".bat", ".cmd", ".msi", ".dll", ".jar", ".apk"
}


# Words commonly used in phishing URLs to trick users into entering data
SUSPICIOUS_KEYWORDS = {
    "login", "signin", "verify", "verification",
    "account", "secure", "security", "password",
    "payment", "confirm", "credential", "authenticate",
    "authentication", "update", "wallet", "billing",
    "recover", "suspended", "unlock"
}


def calculate_entropy(value):
    # Empty text has no randomness
    if not value:
        return 0.0

    # Count how often each character appears
    counts = Counter(value)
    length = len(value)
    entropy = 0.0

    # Shannon entropy: -sum(p * log2(p)) over all characters.
    # Higher value = more random-looking text.
    for count in counts.values():
        probability = count / length
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy


def safe_port(parsed):
    # Reading .port can raise an error for invalid ports (e.g. "host:abc"),
    # so catch it and return None instead of crashing
    try:
        return parsed.port
    except (ValueError, TypeError):
        return None


def normalize_url(url):
    # Missing value -> empty string
    if url is None:
        return ""

    # Convert to text and strip surrounding spaces
    url = str(url).strip()

    if not url:
        return ""

    # If there is no scheme (http://, https://...), add "http://"
    # so urlparse reads the hostname correctly
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    return url


def is_ip_address(hostname):
    if not hostname:
        return 0

    # IPv6 hosts are wrapped in [ ], so remove the brackets
    hostname = hostname.strip("[]")

    # ip_address() raises ValueError if the text is not a valid IPv4/IPv6
    try:
        ipaddress.ip_address(hostname)
        return 1
    except ValueError:
        return 0


def get_domain_parts(hostname):
    # Returns (domain_name, subdomains, tld)
    if not hostname:
        return "", [], ""

    # Lowercase and remove leading/trailing dots
    hostname = hostname.lower().strip(".")

    # Split on "." and ignore empty pieces
    labels = [
        part for part in hostname.split(".") if part
    ]

    if not labels:
        return "", [], ""

    # Single-label host (e.g. "localhost"): no real TLD
    if len(labels) == 1:
        return labels[0], [], labels[0]

    # Last label = TLD, second-last = domain, everything before = subdomains.
    # Note: this simple split treats "example.co.uk" as domain "co", tld "uk".
    tld = labels[-1]
    domain_name = labels[-2]
    subdomains = labels[:-2]

    return domain_name, subdomains, tld


def tokenize_url(url):
    # Split the URL on every non-alphanumeric character and lowercase it
    tokens = re.split(r"[^a-zA-Z0-9]+", url.lower())

    # Remove empty tokens created by consecutive separators
    return [
        token for token in tokens if token
    ]


def count_special_characters(value):
    if not value:
        return 0

    # Count every character that is not a letter or digit
    return len(
        re.findall(r"[^a-zA-Z0-9]", value)
    )


def longest_digit_run(value):
    if not value:
        return 0

    # Find all groups of consecutive digits
    runs = re.findall(r"\d+", value)

    if not runs:
        return 0

    # Return the length of the longest group
    return max(len(run) for run in runs)


def longest_alpha_run(value):
    if not value:
        return 0

    # Find all groups of consecutive letters
    runs = re.findall(r"[A-Za-z]+", value)

    if not runs:
        return 0

    # Return the length of the longest group
    return max(len(run) for run in runs)


def count_repeated_separators(value):
    if not value:
        return 0

    count = 0

    # Count each run of 2+ identical separators: "--", "__", "..", "//"
    count += len(re.findall(r"-{2,}", value))
    count += len(re.findall(r"_{2,}", value))
    count += len(re.findall(r"\.{2,}", value))
    count += len(re.findall(r"/{2,}", value))

    return count


def count_suspicious_keywords(url):
    if not url:
        return 0

    lower_url = url.lower()

    # Count how many different suspicious keywords appear anywhere in the URL
    # (each keyword counts once, even if it appears several times)
    return sum(
        1 for keyword in SUSPICIOUS_KEYWORDS
        if keyword in lower_url
    )


def calculate_features(url):

    # Clean the input and make sure it has a scheme
    url = normalize_url(url)

    # Parse the URL into parts; fall back to an empty URL if parsing fails
    try:
        parsed = urlparse(url)
    except Exception:
        parsed = urlparse("")

    # Get the hostname (can raise errors on malformed URLs)
    try:
        hostname = parsed.hostname or ""
    except Exception:
        hostname = ""

    hostname = hostname.lower()

    # Other URL parts (empty string if missing)
    path = parsed.path or ""
    query = parsed.query or ""
    fragment = parsed.fragment or ""

    # Split hostname into domain name, subdomains and TLD
    domain_name, subdomains, tld = get_domain_parts(hostname)

    # Word/number tokens of the whole URL
    tokens = tokenize_url(url)

    # --- Basic length features ---
    url_length = len(url)
    domain_length = len(hostname)
    path_length = len(path)
    query_length = len(query)
    fragment_length = len(fragment)

    # --- Digit counts (whole URL, hostname, path, query) ---
    digit_count = sum(
        char.isdigit() for char in url
    )

    domain_digit_count = sum(
        char.isdigit() for char in hostname
    )

    path_digit_count = sum(
        char.isdigit() for char in path
    )

    query_digit_count = sum(
        char.isdigit() for char in query
    )

    # Percentage of URL characters that are digits (guard against divide-by-zero)
    percentage_numeric_chars = (
        digit_count / url_length * 100
        if url_length > 0 else 0.0
    )

    # Share of hostname characters that are digits
    domain_digit_ratio = (
        domain_digit_count / domain_length
        if domain_length > 0 else 0.0
    )

    # --- Hyphens in the hostname ---
    domain_hyphen_count = hostname.count("-")

    domain_hyphen_ratio = (
        domain_hyphen_count / domain_length
        if domain_length > 0 else 0.0
    )

    # Hostname parts separated by dots (e.g. www, example, com)
    domain_labels = [
        part for part in hostname.split(".")
        if part
    ]

    # Length of the longest hostname part (default=0 if there are none)
    max_domain_label_length = max(
        (len(x) for x in domain_labels),
        default=0
    )

    # --- Token features ---
    token_count = len(tokens)

    max_token_length = max(
        (len(token) for token in tokens),
        default=0
    )

    # Count query parameters (keep_blank_values counts "?a=" too)
    try:
        query_parameters = parse_qs(
            query,
            keep_blank_values=True
        )
        query_param_count = len(query_parameters)
    except Exception:
        query_param_count = 0

    # --- TLD features ---
    tld_length = len(tld)

    # 1 if the TLD is in the common list, otherwise 0
    tld_popularity = (
        1 if tld in COMMON_TLDS else 0
    )

    # 1 if the host is an IP address (common in phishing)
    has_ip = is_ip_address(hostname)

    # --- Simple character counts over the whole URL ---
    dot_count = url.count(".")

    hyphen_count = url.count("-")

    special_character_count = (
        count_special_characters(url)
    )

    # --- Path features ---
    # Path parts between slashes, ignoring empty ones
    path_segments = [
        segment for segment in path.split("/")
        if segment
    ]

    path_segment_count = len(path_segments)

    path_special_character_count = (
        count_special_characters(path)
    )

    lower_path = path.lower()

    # Flag URLs that point to risky file types (.exe, .zip, ...)
    suspicious_file_extension = 0

    for extension in SUSPICIOUS_EXTENSIONS:
        if lower_path.endswith(extension):
            suspicious_file_extension = 1
            break

    # Number of phishing-style keywords found in the URL
    suspicious_keyword_count = (
        count_suspicious_keywords(url)
    )

    # --- Entropy (randomness) of URL, hostname and path ---
    url_entropy = calculate_entropy(url)
    domain_entropy = calculate_entropy(hostname)
    path_entropy = calculate_entropy(path)

    # --- Port, "@" and encoding features ---
    port = safe_port(parsed)

    # 1 if the URL contains an explicit port number
    has_port = (
        1 if port is not None else 0
    )

    # "@" can hide the real destination (user@host trick)
    has_at_symbol = (
        1 if "@" in url else 0
    )

    # Count percent-encoded characters like %20 or %2F
    percent_encoded_count = len(
        re.findall(
            r"%[0-9a-fA-F]{2}",
            url
        )
    )

    # 1 if the path contains "//" (can be used to confuse redirects)
    double_slash_path = (
        1 if "//" in path else 0
    )

    # Number of repeated separators in the URL
    repeated_separator_count = (
        count_repeated_separators(url)
    )

    # Longest run of digits / letters in the URL
    consecutive_digit_count = (
        longest_digit_run(url)
    )

    consecutive_alpha_count = (
        longest_alpha_run(url)
    )

    # "xn--" marks internationalized (punycode) domains, sometimes used
    # for look-alike (homograph) attacks
    punycode_flag = (
        1 if "xn--" in hostname else 0
    )

    # Collect every feature in a dictionary (key = feature name)
    features = {
        "url_length": url_length,
        "has_ip_address": has_ip,
        "dot_count": dot_count,
        "url_entropy": url_entropy,
        "token_count": token_count,
        "query_param_count": query_param_count,
        "tld_length": tld_length,
        "path_length": path_length,
        "has_hyphen_in_domain":
            1 if domain_hyphen_count > 0 else 0,
        "number_of_digits": digit_count,
        "tld_popularity": tld_popularity,
        "suspicious_file_extension":
            suspicious_file_extension,
        "domain_name_length": domain_length,
        "percentage_numeric_chars":
            percentage_numeric_chars,
        "hyphen_count": hyphen_count,
        "special_character_count":
            special_character_count,
        "path_segment_count": path_segment_count,
        "suspicious_keyword_count":
            suspicious_keyword_count,
        "domain_entropy": domain_entropy,
        "path_entropy": path_entropy,
        "domain_digit_count":
            domain_digit_count,
        "domain_digit_ratio":
            domain_digit_ratio,
        "domain_hyphen_count":
            domain_hyphen_count,
        "domain_hyphen_ratio":
            domain_hyphen_ratio,
        "max_domain_label_length":
            max_domain_label_length,
        "max_token_length":
            max_token_length,
        "query_length": query_length,
        "fragment_length": fragment_length,
        "has_port": has_port,
        "has_at_symbol": has_at_symbol,
        "percent_encoded_count":
            percent_encoded_count,
        "double_slash_path":
            double_slash_path,
        "repeated_separator_count":
            repeated_separator_count,
        "consecutive_digit_count":
            consecutive_digit_count,
        "consecutive_alpha_count":
            consecutive_alpha_count,
        "punycode_flag": punycode_flag,
        "path_digit_count": path_digit_count,
        "path_special_character_count":
            path_special_character_count,
        "query_digit_count": query_digit_count,
        "query_special_character_count":
            count_special_characters(query),
    }

    # Safety check: the dictionary must contain exactly the expected features
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"Feature count mismatch. "
            f"Expected {len(FEATURE_NAMES)}, "
            f"got {len(features)}"
        )

    return features


# Quick self-test: runs only when this file is executed directly
if __name__ == "__main__":

    print("=" * 60)
    print("FEATURE EXTRACTOR TEST")
    print("=" * 60)

    print("Feature count:", len(FEATURE_NAMES))

    # Sample URLs: a mix of normal sites and phishing-looking ones
    test_urls = [
        "https://docs.python.org/",
        "https://www.microsoft.com/",
        "https://www.wikipedia.org/",
        "https://www.targetteddomainurl.com/library/aboutus.html",
        "https://secure-login-example.com/login/verify",
        "http://192.168.1.1/login",
    ]

    # Extract features for each URL and report success or failure
    for test_url in test_urls:

        try:
            result = calculate_features(test_url)

            print()
            print("URL:", test_url)
            print("Extracted features:", len(result))

        except Exception as error:

            print()
            print("ERROR:", test_url)
            print("Reason:", error)