# services/live_url_checker.py

# --- Standard library imports ---
import ipaddress   # Check / classify IP addresses (public vs private)
import socket      # DNS lookups (resolve hostnames to IPs)

# --- Third-party import ---
import requests    # Send HTTP requests to the website being checked

from urllib.parse import urljoin, urlsplit   # Build/split URLs (handles redirects)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_URL_LENGTH = 2048          # Reject URLs longer than this
MAX_REDIRECTS = 5              # Follow at most 5 redirects
REQUEST_TIMEOUT = 10           # Seconds to wait for the website
RDAP_TIMEOUT = 8               # Seconds to wait for domain-registration lookups
MAX_RESPONSE_BYTES = 1_000_000 # Read at most ~1 MB of a page (protects memory)

# Browser-like User-Agent so sites respond as they would to a normal visitor
USER_AGENT = (
    "Mozilla/5.0 "
    "(Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "Chrome/154.0 Safari/537.36"
)

# Phrases commonly found on phishing pages (matched in lowercase HTML)
SUSPICIOUS_WORDS = [
    "verify your account",
    "confirm your account",
    "login immediately",
    "verify your identity",
    "account suspended",
    "enter your password",
    "security verification",
    "confirm your password",
]


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):
    """Normalize a URL without silently accepting invalid input."""

    # Only text is accepted
    if not isinstance(url, str):
        return ""

    url = url.strip()

    # Reject empty or overly long URLs
    if not url or len(url) > MAX_URL_LENGTH:
        return ""

    try:
        parsed = urlsplit(url)

        # No scheme given -> assume HTTPS and parse again
        if not parsed.scheme:
            url = "https://" + url
            parsed = urlsplit(url)

        # Only web URLs are allowed (blocks file://, ftp://, javascript:, etc.)
        if parsed.scheme.lower() not in ("http", "https"):
            return ""

        # A URL must have a hostname
        if not parsed.hostname:
            return ""

        # Reject URLs with embedded credentials (user:pass@host)
        if parsed.username is not None or parsed.password is not None:
            return ""

        # Accessing .port validates malformed port syntax.
        port = parsed.port

        # Port must be within the valid range
        if port is not None and not (1 <= port <= 65535):
            return ""

        return url

    except (ValueError, TypeError):
        # Any parsing error -> invalid URL
        return ""


def get_domain(url):
    """Return the normalized hostname, or an empty string."""

    # Validate / clean the URL first
    normalized = normalize_url(url)

    if not normalized:
        return ""

    try:
        parsed = urlsplit(normalized)
        # Lowercase the host and remove any trailing dot
        return (parsed.hostname or "").lower().rstrip(".")

    except (ValueError, TypeError):
        return ""


# ============================================================
# PUBLIC DESTINATION VALIDATION
# ============================================================
# These functions protect the server from SSRF: tricking it into
# requesting internal addresses (localhost, 192.168.x.x, etc.).

def is_public_ip_address(address):
    """Return True only for globally routable IP addresses."""

    try:
        ip = ipaddress.ip_address(address)

        # Also reject IPv4-mapped IPv6 addresses if their embedded
        # IPv4 address is not globally routable.
        if isinstance(ip, ipaddress.IPv6Address):
            if ip.ipv4_mapped is not None:
                return ip.ipv4_mapped.is_global

        # is_global is False for private, loopback, link-local, etc.
        return ip.is_global

    except ValueError:
        # Not a valid IP address
        return False


def resolve_public_addresses(hostname, port):
    """
    Resolve a hostname and reject it if any returned address
    is not globally routable.

    Returns a list of IP address strings, or an empty list.
    """

    try:
        # Ask DNS for all TCP addresses of this host
        results = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
        )

        addresses = set()

        # Collect the IP address from each DNS result
        for result in results:
            sockaddr = result[4]

            if not sockaddr:
                return []

            addresses.add(sockaddr[0])

        if not addresses:
            return []

        # Reject mixed public/private DNS answers.
        if not all(is_public_ip_address(addr) for addr in addresses):
            return []

        return list(addresses)

    except (socket.gaierror, OSError, ValueError):
        # DNS failure -> treat as unsafe / unresolved
        return []


def is_safe_public_url(url):
    """
    Allow only HTTP(S) URLs pointing to public destinations.

    Note: this check alone cannot completely prevent DNS rebinding,
    because a separate HTTP client connection can resolve the
    hostname again.
    """

    # Basic input checks
    if not isinstance(url, str) or not url:
        return False

    if len(url) > MAX_URL_LENGTH:
        return False

    try:
        parsed = urlsplit(url)

        # Only HTTP / HTTPS
        if parsed.scheme.lower() not in ("http", "https"):
            return False

        if not parsed.hostname:
            return False

        # No embedded credentials
        if parsed.username is not None or parsed.password is not None:
            return False

        port = parsed.port

        # Port must be valid
        if port is not None and not (1 <= port <= 65535):
            return False

        hostname = parsed.hostname.rstrip(".").lower()

        if not hostname:
            return False

        # Block local-only hostnames
        if (
            hostname == "localhost"
            or hostname.endswith(".localhost")
            or hostname.endswith(".local")
        ):
            return False

        # IP literal: check directly without DNS.
        try:
            ipaddress.ip_address(hostname)
            return is_public_ip_address(hostname)

        except ValueError:
            # Not an IP address -> it is a normal hostname, continue
            pass

        # Use the default port for the scheme if none was given
        effective_port = port or (
            443 if parsed.scheme.lower() == "https" else 80
        )

        # Resolve the hostname; all returned IPs must be public
        addresses = resolve_public_addresses(
            hostname,
            effective_port,
        )

        return bool(addresses)

    except (ValueError, TypeError, OSError):
        return False


# ============================================================
# SAFE HTTP REQUESTS WITH REDIRECT VALIDATION
# ============================================================

def safe_get_with_redirect_checks(url):
    """
    Request a URL without automatic redirects.

    Every redirect target is validated before the next request.
    Raises ValueError for unsafe destinations or excessive redirects.

    This does not completely eliminate DNS rebinding because
    requests may perform a new DNS resolution when connecting.
    """

    current_url = normalize_url(url)

    if not current_url:
        raise ValueError("Invalid URL.")

    # Original request plus up to MAX_REDIRECTS redirects
    for redirect_count in range(MAX_REDIRECTS + 1):

        # Validate each destination before connecting.
        if not is_safe_public_url(current_url):
            raise ValueError(
                "Blocked unsafe URL or redirect destination."
            )

        # allow_redirects=False so we can check every hop ourselves;
        # stream=True so the body is not downloaded all at once
        response = requests.get(
            current_url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=False,
            stream=True,
            headers={
                "User-Agent": USER_AGENT,
            },
        )

        # Not a redirect status -> this is the final response
        if response.status_code not in (301, 302, 303, 307, 308):
            return response

        # It is a redirect: read where it points, then free the connection
        location = response.headers.get("Location")
        response.close()

        if not location:
            raise ValueError("Redirect has no Location header.")

        # Stop if we have already followed the maximum number of redirects
        if redirect_count >= MAX_REDIRECTS:
            raise ValueError("Too many redirects.")

        # Supports relative and absolute redirect locations.
        current_url = urljoin(current_url, location)

        if len(current_url) > MAX_URL_LENGTH:
            raise ValueError("Redirect URL exceeds the length limit.")

    raise ValueError("Redirect limit exceeded.")


# ============================================================
# DOMAIN REGISTRATION CHECK
# ============================================================

def check_domain_registration(domain):
    """
    Check domain registration through IANA RDAP bootstrap data.

    True  = registered
    False = not registered
    None  = unknown

    This is supporting information only. It is not used to
    determine whether a URL is phishing.
    """

    if not domain:
        return None

    try:
        # IANA publishes which RDAP server handles each TLD
        bootstrap_url = "https://data.iana.org/rdap/dns.json"

        response = requests.get(
            bootstrap_url,
            timeout=RDAP_TIMEOUT,
            headers={
                "User-Agent": "PhishGuard/1.0",
            },
        )

        try:
            # Could not get the bootstrap list -> unknown
            if response.status_code != 200:
                return None

            data = response.json()

        finally:
            # Always release the connection
            response.close()

        # TLD = last part of the domain (e.g. "com")
        tld = domain.split(".")[-1].lower()
        rdap_server = None

        # Find the RDAP server responsible for this TLD
        for service in data.get("services", []):
            if len(service) < 2:
                continue

            tlds = service[0]       # list of TLDs this entry covers
            servers = service[1]    # list of RDAP server URLs

            if tld in [str(item).lower() for item in tlds]:
                if servers:
                    rdap_server = servers[0]
                    break

        # No RDAP server for this TLD -> unknown
        if not rdap_server:
            return None

        # Domain names are normalized before constructing the path.
        rdap_url = (
            rdap_server.rstrip("/")
            + "/domain/"
            + domain
        )

        # Ask the RDAP server about this specific domain
        rdap_response = requests.get(
            rdap_url,
            timeout=RDAP_TIMEOUT,
            headers={
                "Accept": "application/rdap+json",
                "User-Agent": "PhishGuard/1.0",
            },
        )

        try:
            # 200 = record found -> registered
            if rdap_response.status_code == 200:
                return True

            # 404 / 410 = no record -> not registered
            if rdap_response.status_code in (404, 410):
                return False

            # Anything else (rate limit, server error...) -> unknown
            return None

        finally:
            rdap_response.close()

    except (
        requests.RequestException,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
    ):
        # Any network / parsing problem -> unknown
        return None


# ============================================================
# DNS CHECK
# ============================================================

def check_dns(domain):
    """
    Return True if the domain resolves, False if resolution fails,
    or None if the result is unknown.

    This is a basic DNS check, not a complete SSRF security control.
    """

    if not domain:
        return None

    try:
        # Succeeds if the domain has at least one IP address
        socket.gethostbyname(domain)
        return True

    except socket.gaierror:
        # Domain does not resolve
        return False

    except (OSError, ValueError):
        # Other errors -> cannot tell
        return None


# ============================================================
# RESPONSE CONTENT ANALYSIS
# ============================================================

def calculate_live_evidence(html):
    """Calculate the existing simple live-page evidence score."""

    live_score = 0

    # Page contains a form (could collect data)
    if "<form" in html:
        live_score += 10

    # Page asks for a password
    if 'type="password"' in html:
        live_score += 20

    # Page embeds an iframe (sometimes used to hide content)
    if "<iframe" in html:
        live_score += 10

    # Any suspicious phrase adds 20 points, only once
    for word in SUSPICIOUS_WORDS:
        if word in html:
            live_score += 20
            break

    # Cap the score at 100
    live_score = min(live_score, 100)

    # Convert the score into a simple risk label
    if live_score >= 60:
        assessment = "High"
    elif live_score >= 30:
        assessment = "Medium"
    else:
        assessment = "Low"

    return live_score, assessment


def read_limited_response(response):
    """
    Read no more than MAX_RESPONSE_BYTES from the response body.
    """

    chunks = []
    total_bytes = 0

    # Download the body in 8 KB pieces
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue

        # How many bytes we are still allowed to read
        remaining = MAX_RESPONSE_BYTES - total_bytes

        if remaining <= 0:
            break

        # Trim the last chunk so we never exceed the limit
        chunk = chunk[:remaining]
        chunks.append(chunk)
        total_bytes += len(chunk)

        if total_bytes >= MAX_RESPONSE_BYTES:
            break

    # Join all pieces into one bytes object
    content = b"".join(chunks)

    # Use the encoding the server reported, default to UTF-8
    encoding = response.encoding or "utf-8"

    # errors="replace" avoids crashes on invalid characters
    return content.decode(encoding, errors="replace")


# ============================================================
# MAIN LIVE URL CHECK
# ============================================================

def check_live_url(url):
    """
    Existing function expected by prediction_service.py.

    Return keys remain compatible with the current application:
      checked
      reachable
      live_score
      live_assessment

    Domain registration information is intentionally not returned.
    """

    # Validate and clean the input URL
    url = normalize_url(url)

    # Invalid URL -> not checked
    if not url:
        return {
            "checked": False,
            "reachable": False,
            "live_score": 0,
            "live_assessment": "Unknown",
        }

    # Extract the hostname
    domain = get_domain(url)

    # No usable hostname -> not checked
    if not domain:
        return {
            "checked": False,
            "reachable": False,
            "live_score": 0,
            "live_assessment": "Unknown",
        }

    # --------------------------------------------------------
    # HIDDEN DOMAIN REGISTRATION CHECK
    # --------------------------------------------------------

    registration_status = check_domain_registration(domain)

    # Keep the result internal, as in the original implementation.
    # Registration status is not proof that a website is safe.
    _ = registration_status

    # --------------------------------------------------------
    # PUBLIC DESTINATION VALIDATION
    # --------------------------------------------------------

    # Refuse to contact private / local / unsafe destinations
    if not is_safe_public_url(url):
        return {
            "checked": True,
            "reachable": False,
            "live_score": 0,
            "live_assessment": "Unknown",
        }

    # --------------------------------------------------------
    # DNS CHECK
    # --------------------------------------------------------

    dns_status = check_dns(domain)

    # Domain does not resolve (or unknown) -> do not request it
    if dns_status is not True:
        return {
            "checked": True,
            "reachable": False,
            "live_score": 0,
            "live_assessment": "Unknown",
        }

    # --------------------------------------------------------
    # WEBSITE REQUEST
    # --------------------------------------------------------

    # Default results, replaced if the request succeeds
    reachable = False
    live_score = 0
    live_assessment = "Unknown"

    response = None

    try:
        # Fetch the page, validating every redirect hop
        response = safe_get_with_redirect_checks(url)

        # Treat unsuccessful HTTP status codes as not reachable
        # for this live-evidence assessment.
        response.raise_for_status()

        # Read a limited amount of HTML and lowercase it for matching
        html = read_limited_response(response).lower()
        reachable = True

        # Score the page content for phishing-like signs
        live_score, live_assessment = calculate_live_evidence(html)

    except (
        requests.RequestException,
        ValueError,
        OSError,
    ):
        # Any network / validation error -> not reachable
        reachable = False
        live_score = 0
        live_assessment = "Unknown"

    finally:
        # Always close the connection
        if response is not None:
            response.close()

    # --------------------------------------------------------
    # REGISTRATION STATUS IS NOT RETURNED TO THE FRONTEND
    # --------------------------------------------------------

    return {
        "checked": True,
        "reachable": reachable,
        "live_score": live_score,
        "live_assessment": live_assessment,
    }


# ============================================================
# BASIC LOCAL TEST
# ============================================================

# Runs only when this file is executed directly
if __name__ == "__main__":
    # Mix of normal, local (should be blocked) and too-long URLs
    test_urls = [
        "https://example.com",
        "https://docs.python.org/",
        "http://127.0.0.1/",
        "http://localhost/",
        "https://example.com/" + ("a" * MAX_URL_LENGTH),
    ]

    # Print whether each URL passes the safe-public-URL check
    for test_url in test_urls:
        print("-" * 60)
        print("URL:", test_url[:120])
        print("Safe public URL:", is_safe_public_url(normalize_url(test_url)))

    print("-" * 60)
    print("Feature check completed.")