import { ImageResponse } from "next/og";

export const size = { width: 32, height: 32 };
export const contentType = "image/png";

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: 32,
          height: 32,
          background: "hsl(217, 91%, 60%)",
          borderRadius: 8,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 21,
          color: "white",
          fontWeight: 700,
          lineHeight: 1,
          paddingBottom: 1,
        }}
      >
        ₿
      </div>
    ),
    { ...size }
  );
}
