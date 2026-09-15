# Annotation guideline — GWRHelp intents (v1.0)

This is the document every golden-set label was assigned against. It is
generated from `artifacts/taxonomy_final.json`; edit the JSON, not this file.

Labelling rules that apply to every intent:

1. Label what the customer **wants**, not the topic they mention.
2. If a message contains several asks, label the one with the highest stakes
   (money > stranded > factual question > venting).
3. Sarcasm does not change intent. Label the grievance underneath it.
4. Do not use information from later turns in the thread. The agent sees only
   the opening message, so the label must be assignable from it alone.
5. If two intents both genuinely fit and rule 2 does not separate them, mark the
   item ambiguous rather than guessing. Ambiguous items are reported, not dropped.

| intent | prior share | needs live data | default disposition |
| --- | --- | --- | --- |
| `live_service_status` | 41.4% | yes | `escalate` |
| `service_complaint` | 8.0% | no | `assist` |
| `compensation_and_refunds` | 9.6% | no | `assist` |
| `ticketing_and_booking` | 10.0% | no | `assist` |
| `seat_reservation` | 2.4% | yes | `assist` |
| `capacity_and_overcrowding` | 7.5% | yes | `assist` |
| `onboard_facilities` | 5.0% | no | `assist` |
| `lost_property` | 4.6% | no | `auto` |
| `missed_connection` | 1.6% | yes | `escalate` |
| `service_information` | 5.6% | no | `assist` |
| `feedback_positive` | 1.8% | no | `auto` |
| `other` | 2.5% | no | `escalate` |

## `live_service_status`

Customer asks for the current or imminent running state of a specific service or route: is it running, how late is it, has it been cancelled, which platform, when will it arrive.

**Includes**
- 'is the 18:21 WMN to CDF running?'
- 'what time will the delayed 07.53 TWY-PAD get to PAD?'
- 'what platform is the 9.37 from Slough?'
- 'any updates on the 18:01 from Yate to Gloucester?'
- 'how are the trains from Thatcham to Paddington this morning?'
- Questions about whether a rail-replacement bus is running now

**Excludes**
- Asking WHY a train was late or cancelled after the fact -> service_complaint
- Asking about a timetable on a future date -> service_information
- Claiming money back for a delay that already happened -> compensation_and_refunds

**Boundary case.** 'Why is the 11.35 Paddington to Reading going so slowly?' reads like a complaint but the customer wants the current status and ETA of a train they are sitting on. Intent is live_service_status. The test is tense and actionability: if the journey is still in progress and the answer would be a fact about right now, it is live_service_status.

**Disposition `escalate`.** A correct answer requires the live running-information feed, which this agent does not have. Generating a plausible arrival time from historical precedent is the single most dangerous failure mode available to this system: it is confident, specific, and wrong. Auto-handling is off for this intent regardless of model confidence.

## `service_complaint`

Customer complains about service quality that has already happened: recurring lateness, a cancellation they want explained, staff conduct, or general anger about the operator. They want acknowledgement and an explanation, not a live fact.

**Includes**
- 'The 08.02 starts from Maidenhead and yet is so frequently late. Why?'
- 'Staff at Reading as sanctimonious as ever'
- 'another outstanding day for the 16:25 Paddington to Oxford'
- Complaints about driver shortages causing repeated delays

**Excludes**
- Asking for money back -> compensation_and_refunds
- Complaining specifically about crowding -> capacity_and_overcrowding
- Complaining about wifi/catering -> onboard_facilities
- Praise -> feedback_positive

**Boundary case.** 'Yet another delayed train Salisbury to Bristol 11.40. Yet another connection missed. When will you sort this out?' contains a complaint AND a missed connection. Missed-connection wins when the customer is currently stranded and needs onward travel; service_complaint wins when they are venting about a pattern. Here the customer is venting -> service_complaint.

**Disposition `assist`.** An apology plus an explanation can be drafted from precedent, but explanations assert causes ('due to a fault at Reading') that we cannot verify. A human must confirm the cause before it is sent. Staff-conduct complaints escalate outright via the router's risk rules.

## `compensation_and_refunds`

Customer wants money: a Delay Repay claim, a refund for a cancelled or unused ticket, compensation for poor service, or an update on a claim already submitted.

**Includes**
- 'how do I claim delay repay for a 45 minute delay'
- 'Soon to be applying for yet another refund'
- 'I submitted a complaint 3 weeks ago and heard nothing'
- 'I want compensation for last night'

**Excludes**
- Asking whether a train is delayed right now -> live_service_status
- Refund because the website failed to complete a purchase -> ticketing_and_booking

**Boundary case.** 'why was the 1706 Pad to Westbury cancelled? I want my money back' is both. Compensation wins: the money request is the actionable ask and carries the higher risk. When a message contains both a status question and a money question, label the money question.

**Disposition `assist`.** The Delay Repay process is public, stable policy, so the procedural answer is genuinely groundable in precedent. But any reply that states an entitlement ('you are due a full refund') is a financial commitment. Draft, never auto-send. The router escalates outright when a specific amount is named.

## `ticketing_and_booking`

Customer needs help buying, using or changing a ticket: website and app failures, ticket machines, advance fares, season tickets, railcards, ticket validity.

**Includes**
- 'why is your website not letting me book train tickets'
- 'what time do you release the next advance tickets?'
- 'are you aware the ticket machine is out of order at Digby & Sowton?'
- 'I have off saver ticket can I get the 15.22 to Kingham?'

**Excludes**
- Refund for a ticket already bought -> compensation_and_refunds
- Seat reservations specifically -> seat_reservation

**Boundary case.** 'Went into Maidenhead station and asked for 7 days to Paddington starting today, they gave me a ticket starting yesterday' looks like a complaint about staff, but the customer wants the ticket fixed. Ticketing wins when the desired outcome is a corrected or usable ticket.

**Disposition `assist`.** Ticket validity rules are public and stable, which makes precedent genuinely informative. Ticket-machine outage reports are safely auto-acknowledgeable. But validity answers ('yes that ticket is valid on that service') have a real cost if wrong -- the customer can be fined -- so they are drafted, not sent.

## `seat_reservation`

Customer asks about reserved seats: a reservation that is missing or not displayed, how to reserve, quiet-coach and unreserved-carriage availability.

**Includes**
- 'Is there an unreserved carriage on the 12:27 from Paddington to Swansea?'
- 'my seat reservations aren't showing on the train'
- 'how do I reserve a seat on an advance ticket'

**Excludes**
- No seats because the train is overcrowded -> capacity_and_overcrowding
- First-class seating and amenities -> onboard_facilities

**Boundary case.** 'no seats at all on 17:50 from Cardiff to Swansea. Why?' is overcrowding, not reservations -- the customer has nowhere to sit, not a reservation problem. Reservation intent requires a reference to a booking or the reservation system.

**Disposition `assist`.** Whether a specific service has unreserved coaches today depends on the formation actually running, which we cannot see. The general process for reserving is answerable from precedent.

## `capacity_and_overcrowding`

Customer reports or complains that a service is too crowded or too short: standing room only, missing carriages, short formation, people left behind on the platform.

**Includes**
- 'squashed on like sardines on the 4:41'
- 'did the depot misplace the other 2 carriages for the 0632?'
- 'people being turned away AGAIN'
- 'no seats, absolutely rammed'

**Excludes**
- A missing personal seat reservation -> seat_reservation
- General lateness complaints -> service_complaint

**Boundary case.** 'Hi did the depot misplace the other 2 carriages for the 0632 Didcot to Paddington? #searchpartyneeded' is sarcastic but is a genuine short-formation report. Sarcasm does not change intent.

**Disposition `assist`.** Explaining why a specific train was short-formed requires today's fleet availability. An empathetic acknowledgement is draftable, but GWR's real replies often commit to operational facts ('more carriages than usual are undergoing maintenance') that we cannot verify.

## `onboard_facilities`

Customer asks about or complains about facilities on the train: wifi, power sockets, catering and buffet, toilets, first-class amenities, air conditioning.

**Includes**
- 'no wifi since 1845 service left london'
- 'my croissant was rock solid'
- 'paid for first class... train hasnt got first class'
- 'is the buffet open on the 17:00?'

**Excludes**
- Lost belongings -> lost_property
- Crowding -> capacity_and_overcrowding

**Boundary case.** 'will the 14:39 London Paddington be a class 43?' is a rolling-stock enthusiast question, not a facilities complaint. Enthusiast questions about train classes go to service_information unless the customer is asking because of an amenity (e.g. 'will it have first class?').

**Disposition `assist`.** Apology and the refund route for a paid-for amenity that was unavailable are both well-precedented. Whether the buffet is open on a specific service today is not.

## `lost_property`

Customer has left an item on a train or at a station and wants to recover it.

**Includes**
- 'left my north face jacket on the 8.01 from Bristol to Paddington. How do I go about getting it back?'
- 'my colleague left his coat on the 1606 Paddington-Penzance (coach B)'
- 'I left my hat on the 17:55 Newbury to Reading'

**Excludes**
- Damaged or stolen property -> escalate via other
- Lost tickets -> ticketing_and_booking

**Boundary case.** 'someone took my bag off the luggage rack' is theft, not lost property. Theft escalates to a human and is labelled other.

**Disposition `auto`.** The single best auto-handling candidate in this taxonomy. The answer is a fixed, public procedure (report via the lost-property form, quoting service and coach), it does not depend on live data, it does not commit money, and the historical replies are highly consistent. This intent was buried inside a mixed cluster by the automatic consolidation and recovered by hand -- see curation_log C-3.

## `missed_connection`

Customer has missed, or is about to miss, an onward connection because of a GWR delay, and needs to know what to do now.

**Includes**
- 'my connection leaves Westbury at 11:11. What am I meant to do?'
- 'Connection at Slough now missed'
- 'I missed my connection and was stranded at Parkway so sorted a taxi'

**Excludes**
- Claiming the taxi fare back afterwards -> compensation_and_refunds
- Venting about connections in general -> service_complaint

**Boundary case.** 'thanks for the late train from Bath to Bristol, meaning i have missed my connection to Manchester' is sarcastic thanks, a missed connection, and an implied compensation claim. Missed-connection wins while the customer is still travelling, because the time-critical need is onward travel.

**Disposition `escalate`.** A stranded passenger is time-critical and the correct answer depends on live onward services and ticket-acceptance arrangements with other operators. Low volume, high harm if mishandled. Always escalate.

## `service_information`

Customer asks about planned or general service facts rather than today's running: future timetables, engineering work, strike days, new trains and rolling stock, route and station facts.

**Includes**
- 'Are you planning on striking tomorrow?'
- 'how many of the new trains are now in operation?'
- 'when are the first timetabled class 800 services due to begin?'
- 'will there be any HSTs left in service in mid December?'

**Excludes**
- Whether a specific train is running right now -> live_service_status
- Buying a ticket for a future journey -> ticketing_and_booking

**Boundary case.** 'Are the trains to Cardiff affected tomorrow?' is service_information if it concerns published engineering work or a strike, but live_service_status if it concerns disruption already underway that may continue. Default to service_information when the date is not today.

**Disposition `assist`.** Published timetables and announced engineering work are stable and precedent is informative, but the corpus is from 2017 and any specific date-bound fact in a precedent is stale. Drafting is safe; sending is not.

## `feedback_positive`

Customer praises the service, thanks staff, or shares a positive experience, with no question to answer.

**Includes**
- 'THANK U Adrian A27 Amazing just the best service laughs all the way!!!'
- 'New train for the first time. Very exciting.'
- 'both drivers spotted us and gave a little wave'

**Excludes**
- Sarcastic thanks -> the underlying complaint intent
- Thanks that closes out a resolved issue and asks something further -> the further question's intent

**Boundary case.** 'Thanks for the late train from Bath, meaning I missed my connection' is sarcasm. Positive feedback requires genuine praise; if any grievance is present, label the grievance.

**Disposition `auto`.** No factual claim is required, no money is at stake, and the historical replies are short and formulaic. Safe to auto-send. The only real risk is misreading sarcasm as praise, which the router guards with a sentiment check.

## `other`

Anything that does not fit the above: unrelated chatter, messages too vague to act on, non-English messages, theft or injury reports, media and business enquiries, and follow-ups on an existing DM case with no restatable content.

**Includes**
- 'can you help please? <url>'
- 'Don't fancy replying then?'
- 'hi, I've sent a DM, is there any update on my query?'
- Messages in Welsh or other non-English languages

**Excludes**
- Anything that clearly matches a named intent, even if badly written

**Boundary case.** A vague message with an image attached ('can you help please? <url>') is other, because the actionable content is in an image this text-only system cannot read. That is a capability limit, not a classification problem, and it should escalate.

**Disposition `escalate`.** By construction this is the bucket of things the system does not understand. Escalating is the only defensible default, and the size of this bucket is a headline quality metric in its own right.

## Curation log

### C-1 — Recovered two clusters the consolidation silently dropped.

The model's output asserted 'No clusters were dropped as noise; all were assigned to an intent or the catch-all', but clusters 2 (first class / rolling stock, 2.5%) and 3 (missed connections, 1.6%) appeared in no intent's source_clusters. 4.1% of traffic was unassigned while the model reported full coverage. Found by a coverage assertion, not by reading the output.

*Why it matters.* A self-reported completeness claim from an LLM is not evidence. This is now enforced by a test (tests/test_taxonomy.py::test_every_cluster_is_assigned).

### C-2 — Shrank the catch-all from 17.7% to 2.5%.

The consolidation put cluster 9.3 (12.7% of traffic, flagged incoherent) into 'other'. Reading its exemplars showed it is dominated by live departure queries, so it was reassigned to live_service_status. An 'other' bucket holding a sixth of all traffic hides the system's real coverage.

*Why it matters.* Escalation rate is the metric the business pays for. A bloated catch-all that defaults to escalate would have made the agent look safe while being useless.

### C-3 — Split out lost_property as its own intent.

Cluster 9.0 (4.6%) was named 'multiple_intents' and mixed lost property with wifi complaints and delay queries. Lost property is the most auto-handleable intent in the entire corpus -- fixed public procedure, no live data, no money -- and the automatic taxonomy would have buried it in an escalate-by-default bucket.

*Why it matters.* This is the difference between a system with a real auto-handle rate and one with almost none. The clustering could not see it because lost-property messages share vocabulary (service times, station names) with delay queries.

### C-4 — Split live_service_status from service_complaint, and both from compensation_and_refunds.

Kept from the model's proposal and reinforced. 'Is the 18:03 delayed?', 'why is the 18:03 always delayed?' and 'I want money back for the 18:03' share almost all their vocabulary and have three different correct handlings.

*Why it matters.* These three are the highest-volume confusions in the corpus and the confusion matrix in the report is dominated by them.

### C-5 — Overrode live_service_status disposition from 'assist' to 'escalate'.

The model proposed assist. At 41% of traffic this is the largest intent, and a drafted reply containing an invented arrival time is more dangerous than no reply, because a human reviewer under time pressure is likely to send a fluent draft without checking the number.

*Why it matters.* Automation bias is a real operational risk, not a hypothetical. Drafting plausible-looking false facts is worse than abstaining.

### C-6 — Replaced all LLM-estimated shares with shares derived from cluster sizes, and renamed the field to prior_share.

The model's est_share values summed to 71.6%, not 100%. They are now computed as the sum of source-cluster shares and asserted to sum to 1.0 in tests.

*Why it matters.* An intent-distribution table is exactly the kind of number a reader trusts without checking. It must not come from a language model's arithmetic.
