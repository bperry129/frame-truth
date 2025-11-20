export default function handler(req, res) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', '*');
  
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  try {
    // Get IP from various headers
    const xForwardedFor = req.headers['x-forwarded-for'] || '';
    const xRealIp = req.headers['x-real-ip'] || '';
    const cfConnectingIp = req.headers['cf-connecting-ip'] || '';
    const vercelForwardedFor = req.headers['x-vercel-forwarded-for'] || '';
    
    // Extract the first IP from x-forwarded-for
    let clientIp = xForwardedFor.split(',')[0].trim();
    
    // If still empty, try other headers
    if (!clientIp) {
      clientIp = xRealIp || cfConnectingIp || vercelForwardedFor || req.connection?.remoteAddress || 'unknown';
    }
    
    // Whitelist check
    const whitelistedIps = ["127.0.0.1", "localhost", "::1", "0.0.0.0", "192.168.1.16", "173.239.214.13"];
    const isWhitelisted = whitelistedIps.includes(clientIp) || 
                         clientIp.startsWith('127.') || 
                         clientIp.startsWith('192.168.') || 
                         clientIp.startsWith('10.') || 
                         clientIp.startsWith('172.');
    
    const responseData = {
      detected_ip: clientIp,
      client_connection: req.connection?.remoteAddress || 'unknown',
      x_forwarded_for: xForwardedFor,
      x_real_ip: xRealIp,
      cf_connecting_ip: cfConnectingIp,
      vercel_forwarded_for: vercelForwardedFor,
      is_whitelisted: isWhitelisted,
      all_headers: req.headers,
      method: req.method,
      url: req.url
    };
    
    res.status(200).json(responseData);
    
  } catch (error) {
    console.error('IP check error:', error);
    res.status(500).json({ 
      error: 'Internal server error', 
      message: error.message,
      stack: error.stack 
    });
  }
}
