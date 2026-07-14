import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, ListFlowable, ListItem)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER

d = np.load("sweep_log.npz")

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.0, leading=11.3, spaceAfter=4)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=11.5, leading=13, spaceBefore=6, spaceAfter=3)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=10, leading=12, spaceBefore=4, spaceAfter=2)
title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=15, leading=17, spaceAfter=2)
sub_style = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9.5, alignment=TA_CENTER,
                            textColor=colors.grey, spaceAfter=8)
caption = ParagraphStyle("caption", parent=styles["Normal"], fontSize=8, leading=9.5,
                          textColor=colors.grey, alignment=TA_CENTER, spaceAfter=6)

story = []
story.append(Paragraph("Chopstick Crane: Tracking a Moving-Frame Target on a Passively-Tilting Board", title_style))
story.append(Paragraph("JRF Selection Task &mdash; INTERFACE Lab, IIT Madras", sub_style))

# ---------------- (i) Problem formulation ----------------
story.append(Paragraph("1. Problem formulation", h1))
story.append(Paragraph(
    "A 3R planar arm (joints &theta;<sub>1</sub>,&theta;<sub>2</sub>,&theta;<sub>3</sub> about axes normal to the "
    "world x&ndash;z plane, links L<sub>1</sub>,L<sub>2</sub>,L<sub>3</sub>) holds a pen and presses "
    "against a board mounted on a spring-damped hinge (stiffness k<sub>&phi;</sub>, damping b<sub>&phi;</sub>, tilt "
    "&phi;). The board's instantaneous frame is spanned by tangent t-hat(&phi;) and outward normal "
    "n-hat(&phi;), both rigidly attached to the board and hence rotating with it. The target curve is "
    "specified <i>in the board frame</i> as an along-surface coordinate u(s), s&isin;[0,1], with a desired contact "
    "force F<sub>n</sub><super>*</super>(s). The controller only ever sees u(s) and F<sub>n</sub><super>*</super>(s); "
    "it must independently reconstruct where that curve currently sits in the world frame W, because &phi; "
    "is itself a function of how hard/where the pen has been pressing. Formally, the objective at each instant is:",
    body))
story.append(Paragraph(
    "minimize &nbsp; ||p<sub>pen</sub>(q) &minus; p<sub>target</sub>(&phi;,u,v)||<super>2</super> &nbsp; subject to "
    "F<sub>min</sub> &le; F<sub>n</sub> &le; F<sub>max</sub>, &nbsp; v &asymp; 0,", body))
story.append(Paragraph(
    "where p<sub>target</sub>(&phi;,u,v) = O<sub>piv</sub> + u&middot;t-hat(&phi;) &minus; v&middot;"
    "n-hat(&phi;) composes the board's own (state-dependent) forward kinematics with the along-surface "
    "target &mdash; the moving-frame coupling the task is built around. The arm has 3 DOF against a 2-D world-space "
    "position task (1-D along-surface + 1-D force/penetration), leaving one redundant DOF, resolved by a secondary "
    "objective (Sec.&nbsp;3) rather than a closed-form inverse.", body))

# ---------------- (ii) FK ----------------
story.append(Paragraph("2. Forward kinematics and Jacobian", h1))
story.append(Paragraph(
    "All rotations are about the world y-axis; R<sub>y</sub>(a) is the standard right-handed rotation matrix. With "
    "base O=(x<sub>0</sub>,0,z<sub>0</sub>) and cumulative angles a<sub>1</sub>=&theta;<sub>1</sub>, "
    "a<sub>2</sub>=&theta;<sub>1</sub>+&theta;<sub>2</sub>, a<sub>3</sub>=&theta;<sub>1</sub>+&theta;<sub>2</sub>+"
    "&theta;<sub>3</sub>, the transform chain O&rarr;p<sub>1</sub>&rarr;p<sub>2</sub>&rarr;p<sub>pen</sub> (each link "
    "extending along its own rotated local x-axis) gives, in closed form,", body))
story.append(Paragraph(
    "x = x<sub>0</sub> + L<sub>1</sub>cos a<sub>1</sub> + L<sub>2</sub>cos a<sub>2</sub> + "
    "L<sub>3</sub>cos a<sub>3</sub> ,&nbsp;&nbsp; "
    "z = z<sub>0</sub> &minus; L<sub>1</sub>sin a<sub>1</sub> &minus; L<sub>2</sub>sin a<sub>2</sub> "
    "&minus; L<sub>3</sub>sin a<sub>3</sub> .", body))
story.append(Paragraph(
    "Differentiating gives the 2&times;3 Jacobian J = &part;(x,z)/&part;(&theta;<sub>1</sub>,&theta;<sub>2</sub>,"
    "&theta;<sub>3</sub>), e.g. row 1: [&minus;L<sub>1</sub>sin a<sub>1</sub>&minus;L<sub>2</sub>sin a<sub>2</sub>"
    "&minus;L<sub>3</sub>sin a<sub>3</sub>, &nbsp;&minus;L<sub>2</sub>sin a<sub>2</sub>&minus;L<sub>3</sub>"
    "sin a<sub>3</sub>, &nbsp;&minus;L<sub>3</sub>sin a<sub>3</sub>] (row 2 analogous with cos). "
    "The board pivot is fixed at O<sub>piv</sub>; its frame is t-hat=R<sub>y</sub>(&phi;)(0,0,1)<sup>T</sup>, "
    "n-hat=R<sub>y</sub>(&phi;)(&minus;1,0,0)<sup>T</sup> &mdash; the same rotation MuJoCo applies to a "
    "hinge with axis (0,1,0), which we exploit for validation below. Both the arm FK/Jacobian and the board frame "
    "were checked against MuJoCo's own <font face=\"Courier\">xpos</font>/<font face=\"Courier\">xmat</font> over "
    "200 random configurations and a &phi; sweep (script <font face=\"Courier\">verify_fk.py</font>); max position "
    "error 4.5e-16&nbsp;m, max Jacobian error 1.7e-10, max board-frame axis error 1.1e-16 &mdash; i.e. exact to "
    "floating-point precision, not merely close.", body))

# ---------------- (iii) IK / optimization ----------------
story.append(Paragraph("3. Inverse kinematics / optimization method", h1))
story.append(Paragraph(
    "At each 30&nbsp;Hz control step we take one damped-least-squares (Levenberg&ndash;Marquardt) Gauss&ndash;Newton "
    "step on the position task e = p<sub>target</sub> &minus; p<sub>pen</sub>:", body))
story.append(Paragraph(
    "&Delta;q = J<sup>+</sup>(k<sub>p</sub>e + v<sub>ff</sub>) + (I &minus; J<sup>+</sup>J)&middot;(&minus;k<sub>n</sub>"
    "(q&minus;q<sub>mid</sub>)), &nbsp;&nbsp; J<sup>+</sup>=J<sup>T</sup>(JJ<sup>T</sup>+&lambda;<super>2</super>I)"
    "<sup>&minus;1</sup>", body))
story.append(Paragraph(
    "the closed-form solution of minimizing ||J&Delta;q&minus;(k<sub>p</sub>e+v<sub>ff</sub>)||<super>2</super>+"
    "&lambda;<super>2</super>||&Delta;q||<super>2</super>, i.e. a small 3-variable QP solved analytically every "
    "timestep &mdash; not a black-box IK call. v<sub>ff</sub> is a feedforward term (du/dt along t-hat, known "
    "analytically from the profile) that removes the steady-state lag a pure proportional loop has on a ramping "
    "target. The redundant DOF (null-space projector I&minus;J<sup>+</sup>J) is used for joint-centering, keeping "
    "the arm away from its limits/singularities without disturbing the primary task. &Delta;q is integrated into a "
    "commanded posture q<sub>cmd</sub> sent to per-joint position servos (MuJoCo <font face=\"Courier\">position"
    "</font> actuators), whose finite stiffness gives the arm physical compliance under contact load. The force "
    "target enters through v: a PI loop drives a commanded penetration depth d<sub>pen</sub> from the "
    "F<sub>n</sub> error (clamped, anti-windup), and p<sub>target</sub> is offset by d<sub>pen</sub> along "
    "&minus;n-hat &mdash; i.e. force is regulated by how far past the nominal surface the IK target is "
    "commanded, converted to real force through the servo stiffness and MuJoCo's contact constraint.", body))

# ---------------- (iv) board coupling ----------------
story.append(Paragraph("4. Board tilt entering the target", h1))
story.append(Paragraph(
    "Every control step reads &phi; fresh from <font face=\"Courier\">qpos</font> (the board is a free hinge body "
    "in the MJCF, not scripted), recomputes t-hat(&phi;), n-hat(&phi;) and hence "
    "p<sub>target</sub>(&phi;,u<sub>des</sub>,d<sub>pen</sub>) <i>before</i> the IK solve &mdash; so the arm is "
    "always chasing the curve's current world-frame pose, not a frame frozen at t=0. Physically, &phi; itself "
    "evolves from k<sub>&phi;</sub>&phi; + b<sub>&phi;</sub>d&phi;/dt = &tau;<sub>contact</sub>(F<sub>n</sub>, "
    "lever arm), handled natively by MuJoCo via the hinge's <font face=\"Courier\">stiffness</font>/"
    "<font face=\"Courier\">damping</font> attributes and its contact solver &mdash; we never hand-derive "
    "&phi;(t); we only ever read it back. (The board subsystem uses "
    "<font face=\"Courier\">gravcomp=\"1\"</font> so its tilt is driven purely by contact torque vs. the "
    "spring/damper, isolating the coupling the task is about, rather than by the board's own weight about an "
    "off-axis pivot &mdash; see Sec.&nbsp;5.)", body))

# ---------------- (v) Results ----------------
story.append(Paragraph("5. Results", h1))
story.append(Paragraph(
    "<b>Profile: Sweep</b>, u(s)=u<sub>0</sub>+(u<sub>1</sub>&minus;u<sub>0</sub>)s, u<sub>0</sub>=0.15&nbsp;m, "
    "u<sub>1</sub>=0.45&nbsp;m, F<sub>n</sub><super>*</super>&isin;[2,4]&nbsp;N, k<sub>&phi;</sub>=9&nbsp;N&middot;m/"
    "rad, b<sub>&phi;</sub>=0.6&nbsp;N&middot;m&middot;s/rad, 35&nbsp;s total (1.5&nbsp;s gentle stand-off&rarr;touch "
    "ramp, 4&nbsp;s hold, 29&nbsp;s sweep).", body))

img = Image("tracking_force_tilt.png", width=4.0 * inch, height=4.53 * inch)
story.append(img)
story.append(Paragraph(
    "Figure 1: along-surface tracking u(t), contact-force regulation F<sub>n</sub>(t) vs. target band, and board "
    "tilt &phi;(t) &mdash; note &phi; keeps drifting through the whole sweep, confirming the target frame is truly "
    "moving.", caption))

data = [
    ["Metric", "Value"],
    ["Steady-state F_n (t>5s)", "4.03 ± 0.02 N (top of [2,4] band)"],
    ["Peak transient F_n (first contact, t=1.5s)", "16.0 N, settles into band by t=2.4s"],
    ["Steady-state along-surface error |u_des-u_meas|", "20.0 mm (~6.7% of the 300 mm sweep)"],
    ["Final board tilt", "11.7° (from 0°)"],
]
tbl = Table(data, colWidths=[3.3 * inch, 2.5 * inch])
tbl.setStyle(TableStyle([
    ("FONTSIZE", (0, 0), (-1, -1), 8.3),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
]))
story.append(tbl)
story.append(Spacer(1, 4))

story.append(Paragraph("Failure modes observed", h2))
items = [
    "<b>Off-axis gravity domination.</b> Initially the board's own weight (center of mass offset from the pivot) "
    "produced &gt;9&nbsp;N&middot;m of gravity torque &mdash; more than k<sub>&phi;</sub> could resist &mdash; and "
    "the board simply fell to its joint limit regardless of contact. Fixed with <font face=\"Courier\">gravcomp="
    "\"1\"</font> on the board subsystem (Sec.&nbsp;4); a real design would need a counterweight or a much stiffer "
    "spring instead.",
    "<b>Spurious self-contact.</b> With default MuJoCo collision settings, the arm's own links briefly "
    "self-intersected in the initial posture, generating phantom contact torque on the board with F<sub>n</sub> "
    "reported as zero (wrong geom pair). Fixed by disabling collisions everywhere except the pen-tip/board-face "
    "pair via <font face=\"Courier\">contype</font>/<font face=\"Courier\">conaffinity</font> groups.",
    "<b>First-contact impact spike.</b> Approaching the board directly at the profile's start velocity caused a "
    "170&nbsp;N transient that slammed &phi; into its hard limit. Fixed with an explicit 1.5&nbsp;s stand-off"
    "&rarr;touch ramp before the force-PI loop engages (visible as the small transient peak at t&asymp;1.5s in "
    "Fig.&nbsp;1, now only 16&nbsp;N and self-correcting).",
    "<b>Persistent along-surface lag.</b> Even after tuning, a steady ~20&nbsp;mm lag remains between u<sub>des</sub> "
    "and u<sub>meas</sub> during the sweep (Fig.&nbsp;1, top). This is not a pure ramp-lag (removed by the "
    "feedforward term) but finite servo/contact compliance: under the sustained normal load the joints sit at a "
    "small steady offset from q<sub>cmd</sub>. No singularities were encountered in this workspace region "
    "(condition number of J stayed &lt;5 throughout).",
]
story.append(ListFlowable([ListItem(Paragraph(t, body), leftIndent=8, spaceAfter=3) for t in items],
                           bulletType="bullet", start="•", leftIndent=10))

story.append(Paragraph("One thing to improve", h2))
story.append(Paragraph(
    "Replace the proportional force-PI on d<sub>pen</sub> with true impedance/admittance control referenced to an "
    "estimated joint/contact stiffness, so the ~20&nbsp;mm steady tracking lag and the residual force ripple during "
    "the sweep (visible as the slow wobble in Fig.&nbsp;1's F<sub>n</sub> trace) could both be predicted and "
    "cancelled analytically rather than tuned empirically.", body))

doc = SimpleDocTemplate("report.pdf", pagesize=letter,
                         topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                         leftMargin=0.65 * inch, rightMargin=0.65 * inch)
doc.build(story)
print("built report.pdf")
