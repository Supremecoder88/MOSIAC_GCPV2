const API_BASE_URL = process.env.MOSAIC_API_URL || "http://127.0.0.1:8080";

type RouteContext = {
  params: Promise<{ path: string[] }>;
};

async function proxyRequest(request: Request, context: RouteContext) {
  try {
    const { path } = await context.params;
    const incomingUrl = new URL(request.url);

    const endpoint = `${API_BASE_URL}/api/v1/${path
      .map(encodeURIComponent)
      .join("/")}${incomingUrl.search}`;

    const hasBody = !["GET", "HEAD"].includes(request.method);
    const body = hasBody ? await request.text() : undefined;

    const upstream = await fetch(endpoint, {
      method: request.method,
      headers: hasBody ? { "Content-Type": "application/json" } : undefined,
      body,
      cache: "no-store",
    });

    const responseBody = await upstream.text();

    return new Response(responseBody, {
      status: upstream.status,
      headers: {
        "Content-Type":
          upstream.headers.get("content-type") || "application/json",
      },
    });
  } catch (error) {
    console.error("MOSAIC API proxy error:", error);

    return Response.json(
      { detail: "Could not connect to the MOSAIC backend. Check the Cloud Run proxy on port 8080." },
      { status: 502 }
    );
  }
}

export const GET = proxyRequest;
export const POST = proxyRequest;
export const PATCH = proxyRequest;