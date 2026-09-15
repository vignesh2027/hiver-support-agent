# Intent taxonomy — GWRHelp

Induced from training-split customer messages by clustering, named and
consolidated by an LLM, then hand-edited. This file is also the annotation
guideline: every golden-set label was assigned by applying these rules.

| intent | share | live data needed | default disposition |
| --- | --- | --- | --- |
| `live_disruption_status` | 1250.0% | yes | assist |
| `punctuality_explanation` | 310.0% | no | auto |
| `ticket_purchase_and_booking` | 930.0% | no | assist |
| `seat_reservation` | 240.0% | yes | assist |
| `capacity_and_overcrowding` | 750.0% | yes | assist |
| `compensation_and_refunds` | 1350.0% | no | assist |
| `service_information_general` | 700.0% | no | auto |
| `onboard_service_and_facilities` | 950.0% | no | assist |
| `positive_feedback` | 180.0% | no | auto |
| `other` | 500.0% | no | escalate |

## `live_disruption_status`

Customer asks for real‑time information about a specific train’s delay, cancellation, driver absence or platform, or wants to know if a service is running now.

**Includes**
- inquire_train_delay
- request_service_status
- inquire_morning_service_status
- check_service_status
- request_current_delay_status
- driver_absence_delay
- request_platform_information

**Excludes**
- request_compensation_for_late_arrival
- inquire_delay_and_compensation
- request_delay_information
- request_delay_info_and_compensation

**Boundary case.** “The 08:02 from Maidenhead is always late, why?” – This is a punctuality explanation (historical pattern) not a live status request, so it belongs to punctuality_explanation.

**Disposition.** `assist` — Answers require up‑to‑date running data that the bot cannot guarantee; a human should verify before replying.

## `punctuality_explanation`

Customer seeks an explanation for recurring lateness or systematic delay of a service, not a current status.

**Includes**
- request_punctuality_explanation

**Excludes**
- live_disruption_status
- request_compensation_for_late_arrival

**Boundary case.** “Hi, what time will the delayed 07:53 TWY‑PAD get to PAD?” – This asks for the current expected arrival, so it belongs to live_disruption_status.

**Disposition.** `auto` — A standard template explaining typical causes of lateness can be safely sent.

## `ticket_purchase_and_booking`

Customer needs help buying, printing, refunding, or troubleshooting tickets on the website/app, including season tickets and advance releases.

**Includes**
- ticket_purchase_assistance
- website_booking_issue
- inquire_advance_ticket_release

**Excludes**
- request_seat_reservation
- request_cancellation_info_and_refund

**Boundary case.** “Can I reserve a seat on the 09:15 to Bristol?” – This is a seat reservation request, not a purchase issue, so it belongs to seat_reservation.

**Disposition.** `assist` — May require checking account or availability; a human drafts the response.

## `seat_reservation`

Customer asks to reserve, confirm, change or query a seat reservation on a specific service.

**Includes**
- request_seat_reservation

**Excludes**
- ticket_purchase_and_booking
- request_more_seating_capacity

**Boundary case.** “Why is my train so crowded today?” – This is a capacity complaint, not a reservation query, so it belongs to capacity_requests.

**Disposition.** `assist` — Needs to check reservation system; human assistance required.

## `capacity_and_overcrowding`

Customer complains about overcrowding, lack of carriages or seats and requests additional capacity or fewer passengers.

**Includes**
- request_additional_carriages
- request_more_capacity
- request_more_seating_capacity
- request_more_capacity

**Excludes**
- seat_reservation
- live_disruption_status

**Boundary case.** “Can you add another carriage to the 06:32 Didcot‑Paddington?” – This is a request for extra carriages (capacity), not a seat reservation.

**Disposition.** `assist` — Operational changes need human approval.

## `compensation_and_refunds`

Customer seeks monetary compensation, refund or claim status for a past delay, cancellation or poor service.

**Includes**
- request_compensation_for_late_arrival
- inquire_delay_and_compensation
- request_delay_info_and_compensation
- request_cancellation_info_and_refund
- request_complaint_update
- demand_compensation_for_poor_service

**Excludes**
- live_disruption_status
- punctuality_explanation

**Boundary case.** “Why was the 17:06 Pad‑Westbury cancelled?” – This asks for the reason, not compensation, so it belongs to live_disruption_status.

**Disposition.** `assist` — Compensation requires policy checks and possibly account data.

## `service_information_general`

Customer asks for static or schedule‑related information that does not need real‑time data: new train schedules, future service availability, tomorrow’s service, coach count, or general service queries.

**Includes**
- inquire_new_train_schedule
- request_service_schedule
- inquire_service_tomorrow
- inquire_coach_count
- request_train_time_info

**Excludes**
- live_disruption_status
- ticket_purchase_and_booking

**Boundary case.** “How many coaches will the 07:33 Pang‑Paddington have today?” – If the user asks for today’s count, it needs live data and belongs to live_disruption_status; only future‑date queries stay here.

**Disposition.** `auto` — Answers can be drawn from published timetables.

## `onboard_service_and_facilities`

Customer raises issues or asks questions about onboard amenities, seating comfort, accessibility, Wi‑Fi, toilets, or staff behaviour.

**Includes**
- inquire_onboard_services
- staff_complaint
- request_help

**Excludes**
- capacity_and_overcrowding
- compensation_and_refunds

**Boundary case.** “The Wi‑Fi was unusable, and the train kept getting later.” – This mixes a Wi‑Fi complaint with a delay; the primary intent is a live disruption status, so it belongs to live_disruption_status.

**Disposition.** `assist` — Often needs a human to acknowledge and possibly forward to operations.

## `positive_feedback`

Customer expresses appreciation, compliments or positive experience.

**Includes**
- provide_positive_feedback

**Excludes**
- negative_complaint

**Boundary case.** “Great journey, but the train was 10 minutes late.” – The presence of a complaint pushes it to live_disruption_status.

**Disposition.** `auto` — Can be replied with a thank‑you template.

## `other`

Any message that does not fit the above intents, including spam, unrelated chatter, or ambiguous multi‑intent messages that cannot be cleanly classified.

**Includes**
- multiple_intents
- multiple_unrelated_requests
- noise

**Excludes**

**Boundary case.** “Can I get a replacement taxi pls?” – This is a request for a replacement bus, which belongs to replacement_bus_info; if the bot cannot recognise it, it falls here.

**Disposition.** `escalate` — Human must decide the appropriate handling.

## Consolidation notes

Clusters were merged based on identical handling pathways. All live‑status queries (delays, cancellations, driver absence, platform) became live_disruption_status. All compensation‑related clusters were grouped under compensation_and_refunds because they all require policy lookup and claim processing. Ticket purchase, website issues and advance‑ticket queries share the same support flow (guide, troubleshoot, link) and form ticket_purchase_and_booking. Capacity‑related complaints (extra carriages, overcrowding, standing‑room) were merged into capacity_and_overcrowding. Static schedule queries (new trains, future coach count, tomorrow’s service) formed service_information_general. Onboard amenity complaints and staff behaviour were combined into onboard_service_and_facilities. Positive feedback stayed separate. The catch‑all other captures multi‑intent or noisy messages.

## Dropped clusters

No clusters were dropped as noise; all were assigned to an intent or the catch‑all.
