import './globals.css'
import { Inter, Inclusive_Sans, PT_Serif } from "next/font/google";

const ptSerif = PT_Serif({
  weight: "400",
  style: "italic",
  subsets: ["latin"],
  display: "swap",
  variable: "--font-pt-serif",
});

export const metadata = {
  title: 'Locatr',
  description: 'Locatr Frontend',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={`${ptSerif.variable}`}>{children}</body>
    </html>
  )
}
