# Drone Variant Proposals

## 1. Scout Mech: "The Eye"
*   **Drone Model**: `Spectre` (Scout Class)
*   **Loadout**: 2x Spectre
*   **Role**: Deep Reconnaissance & Electronic Warfare.
*   **Behavior**:
    *   **Flight**: High altitude, fast orbit around enemy concentrations.
    *   **Passive**: High-grade sensor suite (radius 40 voxels) shares vision with team.
    *   **Active**: Emits strong **ECCM** (Counter-Jamming) to clear sensor noise for teammates.
    *   **Stealth**: Harder to lock onto (requires closer range than normal).
*   **Strategic Value**: Essential for breaking enemy "death balls" obscured by jammers.

## 2. Heavy Mech: "The Carrier"
*   **Drone Model**: Hybrid Loadout (1x `Spectre`, 1x `Razor`)
*   **Loadout**: 1x Spectre (Scout), 1x Razor (Combat)
*   **Role**: Independent Hunter-Killer capability.
*   **Launch Mechanic**: **Sequential**.
    1.  **First Activation**: Launches `Spectre`. It flies out to paint/spot targets.
    2.  **Second Activation**: Launches `Razor`.
*   **Razor Drone Specs**:
    *   **Role**: Air Support / Finisher.
    *   **Behavior**: Aggressively seeks targets "Painted" by the Spectre (or any teammate).
    *   **Payload**: **Micro-Missile Pods**. Fires small homing missiles. Low damage individually, but effective at suppressing stability or finishing low-HP stragglers.
*   **Strategic Value**: Allows the Heavy to project force around corners or over terrain without exposing itself.

## 3. Medium Mech: "The Warden" (New Proposal)
*   **Drone Model**: `Aegis` (Support Class)
*   **Loadout**: 2x Aegis
*   **Role**: Mobile Defense & Area Denial.
*   **Behavior**:
    *   **Flight**: Close Tether. Orbits strictly within 5-8 voxels of the owner (or a designated ally?).
    *   **Payload**: **Point Defense Laser (AMS)**.
    *   **Effect**: The drone acts as an external Anti-Missile System. It has its own heat capacity and cooldowns, effectively doubling the Medium mech's missile defense.
    *   **Alt Mode**: If the owner is safe, it can be sent to hover over a capture zone, providing AMS cover for teammates in that zone.
*   **Strategic Value**: Critical for pushing into capture zones under heavy missile fire. A "Shield Bearer" variant.

## 4. Alternative Medium: "The Sentry"
*   **Drone Model**: `Turret-Drone`
*   **Behavior**: Stationary Deployable.
*   **Effect**: Flies to target location and lands/hovers. Acts as a static turret with a Gauss Rifle. High damage, zero mobility once deployed.
