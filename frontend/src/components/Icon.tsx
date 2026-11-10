const PATHS: Record<string, string> = {
  plus: "M12 5v14M5 12h14",
  close: "M6 6l12 12M18 6L6 18",
  chevron: "M6 9l6 6 6-6",
  refresh: "M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7",
  trash: "M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3",
  edit: "M4 20h4L19 9l-4-4L4 16v4zM14 6l4 4",
  menu: "M4 6h16M4 12h16M4 18h16",
  send: "M5 12h14M13 6l6 6-6 6",
  external: "M14 4h6v6M20 4l-9 9M18 14v6H4V6h6",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  check: "M5 12l5 5 9-10",
  code: "M8 8l-4 4 4 4M16 8l4 4-4 4",
  alert: "M12 8v5M12 16.5v.5M10.3 4l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.7-3l-8-14a2 2 0 0 0-3.4 0z",
};

export default function Icon({
  name,
  className = "",
}: {
  name: keyof typeof PATHS | string;
  className?: string;
}) {
  return (
    <svg
      className={`icon ${className}`}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}
