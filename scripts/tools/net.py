import socket

import urllib3.util.connection as urllib3_cn


def pin_ipv4():
    """Force urllib3 (and therefore requests) to resolve hosts as IPv4 only.

    eloratings.net publishes an AAAA record but its IPv6 address black-holes on
    some networks, stalling every request for the full connect timeout before
    falling back to IPv4.
    """
    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
