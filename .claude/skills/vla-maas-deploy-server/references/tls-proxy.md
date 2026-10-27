# Public entry: TLS proxy in front of the gateway

The gateway speaks plain `ws://`/`http://` and listens on `127.0.0.1:9000`. The
proxy terminates TLS on 443 and forwards websocket upgrades. Open only 443 to
the internet. Robots use `https://<domain>` (the client turns it into
`wss://`).

## Caddy (automatic certificates)

```text
# /etc/caddy/Caddyfile
maas.example.com {
    reverse_proxy 127.0.0.1:9000
}
```

Caddy forwards websocket upgrades and keeps idle connections open by default.

## nginx

```nginx
server {
    listen 443 ssl;
    server_name maas.example.com;
    ssl_certificate     /etc/letsencrypt/live/maas.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/maas.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Authorization $http_authorization;
        proxy_read_timeout 3600s;   # a robot session idles between episodes
        proxy_send_timeout 3600s;
        client_max_body_size 16m;
    }
}
```

Without the `Upgrade`/`Connection` headers the robot's handshake fails with an
HTTP error. Without the long read timeout nginx drops sessions after 60 s idle.

## When the cluster cannot open 443

Put the proxy on a host that has a public IP (a jump host or a small cloud VM)
and forward to the gateway over the cluster network, a VPN, or an SSH tunnel:

```bash
# on the public host: forward local 9000 to the gateway node's 9000
ssh -N -L 127.0.0.1:9000:127.0.0.1:9000 user@gateway-node
```

The robot side is unchanged: it only needs the public URL and its key.
