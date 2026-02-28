import './globals.css'

export const metadata = {
  title: 'Locatr',
  description: 'Locatr Frontend',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
