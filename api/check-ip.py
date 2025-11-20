def handler(request):
    """Simple IP checker for Vercel serverless"""
    import json
    
    # Get headers
    headers = dict(request.headers) if hasattr(request, 'headers') else {}
    
    # Extract IP from various sources
    x_forwarded_for = headers.get('x-forwarded-for', '')
    client_ip = x_forwarded_for.split(',')[0].strip() if x_forwarded_for else 'unknown'
    
    # If still unknown, try other headers
    if client_ip == 'unknown':
        client_ip = (headers.get('x-real-ip') or 
                    headers.get('cf-connecting-ip') or 
                    headers.get('x-vercel-forwarded-for') or 
                    'unknown')
    
    # Whitelist check
    whitelisted_ips = ["127.0.0.1", "localhost", "::1", "0.0.0.0", "192.168.1.16", "173.239.214.13"]
    is_whitelisted = (client_ip in whitelisted_ips or 
                     any(client_ip.startswith(prefix) for prefix in ["127.", "192.168.", "10.", "172."]))
    
    response_data = {
        "detected_ip": client_ip,
        "x_forwarded_for": headers.get("x-forwarded-for"),
        "x_real_ip": headers.get("x-real-ip"),
        "cf_connecting_ip": headers.get("cf-connecting-ip"),
        "vercel_forwarded_for": headers.get("x-vercel-forwarded-for"),
        "is_whitelisted": is_whitelisted,
        "all_headers": headers
    }
    
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': '*',
        },
        'body': json.dumps(response_data, indent=2)
    }
