"use client";

import { Bell, Check, CircleHelp, Palette, SlidersHorizontal, UserRound } from "lucide-react";
import { AppShell, Card } from "../../components/app-shell";

export default function Settings() {
  return <AppShell title="Settings" description="Configure the workspace experience and review supported application preferences.">
    <div className="settings-layout">
      <Card className="settings-profile"><div className="settings-section-icon"><UserRound size={18} /></div><div><span className="card-kicker">PROFILE</span><h2>Research profile</h2><p>Neutral workspace profile for this demonstration environment.</p></div><span className="settings-badge"><Check size={13} /> Active</span></Card>
      <div className="settings-grid">
        <Card><div className="settings-heading"><Palette size={17} /><div><span className="card-kicker">APPEARANCE</span><h2>Interface theme</h2></div></div><div className="settings-option selected"><span><strong>Midnight research</strong><small>High-contrast dark workspace surfaces</small></span><Check size={15} /></div><div className="settings-option unavailable"><span><strong>Light theme</strong><small>Not available in this prototype</small></span><span className="coming">Soon</span></div></Card>
        <Card><div className="settings-heading"><SlidersHorizontal size={17} /><div><span className="card-kicker">WORKSPACE</span><h2>Research preferences</h2></div></div><div className="settings-row"><span>Workspace</span><strong>Research Lab</strong></div><div className="settings-row"><span>Inference backend</span><strong className="settings-muted">Not connected</strong></div><div className="settings-row"><span>Default model</span><strong>Attention U-Net</strong></div></Card>
        <Card><div className="settings-heading"><Bell size={17} /><div><span className="card-kicker">NOTIFICATIONS</span><h2>Updates</h2></div></div><div className="settings-option unavailable"><span><strong>Processing notifications</strong><small>Requires a connected backend</small></span><span className="coming">Unavailable</span></div><div className="settings-option unavailable"><span><strong>Research release notes</strong><small>Notification service is not configured</small></span><span className="coming">Unavailable</span></div></Card>
        <Card><div className="settings-heading"><CircleHelp size={17} /><div><span className="card-kicker">APPLICATION</span><h2>About GenVision</h2></div></div><div className="settings-row"><span>Environment</span><strong>Frontend demonstration</strong></div><div className="settings-row"><span>Inference status</span><strong className="settings-muted">Awaiting integration</strong></div><p className="settings-note">No unsupported settings are presented as active controls. Backend inference and persistent preferences will be added when their services are connected.</p></Card>
      </div>
    </div>
  </AppShell>;
}
