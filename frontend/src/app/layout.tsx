import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GenVision — See beyond the clouds",
  description: "AI-powered satellite image reconstruction and agricultural analysis.",
  icons: {
    icon: "/genvision.png",
    apple: "/genvision.png",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
