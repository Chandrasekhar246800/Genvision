"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { BarChart3, Bot, ChevronRight, Cloud, LayoutDashboard, Leaf, Menu, Settings, X, PanelLeftClose, PanelLeftOpen } from "lucide-react";
import { useState } from "react";

const nav = [
  { href: "/dashboard", label: "Overview", icon: LayoutDashboard },
  { href: "/studio", label: "Cloud Removal Studio", icon: Cloud },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
  { href: "/agriculture", label: "Agriculture", icon: Leaf },
  { href: "/models", label: "Model Library", icon: Bot },
];

export function AppShell({ children, title, description }: { children: React.ReactNode; title: string; description?: string }) {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  return <div className="app-shell">
    {open && <button className="sidebar-backdrop" onClick={() => setOpen(false)} aria-label="Close navigation" />}
    <aside className={`app-sidebar ${open ? "open" : ""} ${collapsed ? "collapsed" : ""}`}>
      <div className="sidebar-top"><Link href="/" className="brand"><span className="brand-mark"><Image src="/genvision.png" alt="GenVision" width={27} height={27} priority /></span><span>Gen<span>Vision</span></span></Link><button className="mobile-close" onClick={() => setOpen(false)} aria-label="Close menu"><X size={18} /></button></div>
      <div className="workspace-switch"><span className="workspace-avatar">GV</span><span><small>Workspace</small><strong>Research Lab</strong></span><ChevronRight size={14} /></div>
      <nav className="side-nav"><span className="nav-label">WORKSPACE</span>{nav.map(({ href, label, icon: Icon }) => <Link onClick={() => setOpen(false)} className={pathname === href ? "active" : ""} href={href} key={href}><Icon size={16} /><span>{label}</span>{pathname === href && <span className="nav-active-dot" />}</Link>)}<span className="nav-label nav-label-bottom">SYSTEM</span><Link onClick={() => setOpen(false)} className={pathname === "/settings" ? "active" : ""} href="/settings"><Settings size={16} /><span>Settings</span>{pathname === "/settings" && <span className="nav-active-dot" />}</Link></nav>
      <div className="sidebar-user"><div className="user-avatar">GV</div><div><strong>Research profile</strong><small>Workspace member</small></div><ChevronRight size={14} /></div>
      <button className="sidebar-collapse" onClick={() => setCollapsed(value => !value)} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>{collapsed ? <PanelLeftOpen size={16} /> : <><PanelLeftClose size={16} /><span>Collapse sidebar</span></>}</button>
    </aside>
    <main className="app-main"><header className="app-header"><button className="mobile-menu" onClick={() => setOpen(true)} aria-label="Open menu"><Menu size={20} /></button><div className="breadcrumbs"><span>Research Lab</span><ChevronRight size={13} /><strong>{title}</strong></div><div className="header-actions"><span className="demo-pill"><i /> DEMO MODE</span><div className="header-avatar">GV</div></div></header><div className="app-content"><div className="page-intro"><div><div className="eyebrow">GENVISION / {title.toUpperCase()}</div><h1>{title}</h1>{description && <p>{description}</p>}</div></div>{children}</div></main>
  </div>;
}

export function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) { return <section className={`dash-card ${className}`}>{children}</section>; }
export function Stat({ label, value, change, icon: Icon }: { label: string; value: string; change: string; icon: React.ElementType }) { return <Card className="stat-card"><div className="stat-icon"><Icon size={17} /></div><span className="stat-label">{label}</span><strong className="stat-value">{value}</strong><span className="stat-change">{change}</span></Card>; }
