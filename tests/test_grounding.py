import json

import pytest

from igcs import grounding
from igcs.entities import Doc, Selection


@pytest.mark.parametrize(
    "formatter",
    [
        # Simple case
        json.dumps,
        # IGCS style
        lambda s: f"Selected Content:\n\n{json.dumps(s)}",
        # boxed
        lambda s: f"```json\n{json.dumps(s)}\n```",
    ],
)
def test_grounding(formatter):
    docs = [
        Doc(
            id=0,
            text="UPDATE: Wednesday 5:55 p.m.\nHearne Police say the 93-year-old woman who was shot and killed by one of their officers was armed with a gun.\nAccording to the Robertson County District Attorney's Office, police responded to a 911 call about 6:30 Tuesday evening about a lady waving a gun around, outside a home on Pin Oaks Street.\nNews 3 has learned the person who has called 911 is Pearlie Golden's nephew. A few days ago, the Texas Department of Public Safety refused to renew her driver’s license. The nephew was trying to take her keys away. Golden got upset, grabbed her gun and threatened him.\nWhen Officer Stephen Stem arrived, he asked Golden to “put down the weapon and then ultimately fir(ed) his own sidearm, wounding Miss Golden.”\nMultiple witnesses tell us they heard at least five shots. Golden was pronounced dead at St Joseph's Hospital in Bryan before 10:00 p.m. Tuesday.\nA group of citizens are organizing a candlelight vigil. It will be held tonight at 7:00, in front of Golden's home.\nStem has been placed on paid administrative leave. The Texas Rangers, Hearne Police and the Robertson County District Attorney's Office are investigating. The case will eventually be presented to a grand jury.\n“While it is too early in the investigation to comment on any of the facts or evidence stated above, I can say that this was a very tragic occurrence,” said District Attorney Coty Siegert. “My prayers go out to the family members of Miss Golden and to the Hearne Community as a whole.”\nThis is the second time Stem has been involved in a deadly shooting.\nStem joined the Hearne Police Department on July 22, 2012. Less than six months later, he shot and killed Tederalle Satchell. Police say Satchell was shooting a gun from a car in the Columbus Village Apartments parking lot. A Robertson County Grand Jury did not indict Stem, clearing him of any wrong doing.\nBefore joining Hearne, Stem spent a year with the Lott Police Department and nearly three years with Bryan P.D.\nThe Hearne City Council has called a special meeting for Saturday afternoon. Council members will discuss disciplinary measures for Stem, up to and including termination.\n========\nTuesday 8:30-11:30 p.m.\nA 93-year-old woman is shot and killed by a police officer at her home in Hearne.\nPearlie Golden, known in the neighborhood as Ms. Sully, was pronounced dead Tuesday night at St. Joseph Hospital. The elderly woman was rushed there after being shot by a male officer at her house on Pin Oak Street. Multiple witnesses tell us she was shot at least five times.\nHearne police are not ready to say whether Golden was armed or why the officer felt threatened.\n\"All I know is that they were called out here,” said Robertson County District Attorney Coty Siegert. “They were dispatched out here to address the situation. Again, I'm not sure exactly what that situation was, but it was not a random encounter.\"\nResidents are questioning why police would shoot Golden who they described as a sweet, sweet woman.\n“Even if she did have a gun, she is in her 90’s,” said Lawanda Cooke. “They could have shot in the air to scare her. Maybe she would have dropped it. I don’t see her shooting anyone.\nSiegert says the case will eventually be presented to a Grand Jury, which is standard procedure in officer involved shootings.\nThe Texas Rangers and Robertson County District Attorney’s Office are investigating.\nThe Hearne Police Department says they are working on a news release. We'll bring you that information as soon as it is released.",
            filename=None,
            chunks_pos=None,
            metadata=None,
        ),
        Doc(
            id=1,
            text='(CNN) -- Texas Rangers are investigating why police in a small central Texas town fatally shot a 93-year-old woman at her home.\nPearlie Golden, a longtime resident of Hearne, a town of approximately 4,600 people about 150 miles south of Dallas, was shot multiple times Tuesday.\nA man believed to be a relative of Golden\'s made the 911 call asking for help from police, Robertson County District Attorney Coty Siegert said.\n"What I understand is (Hearne police) were called out because a woman was brandishing a firearm," Siegert said.\n"An officer asked her to put the handgun down, and when she would not, shots were fired."\nHearne City Attorney Bryan Russ Jr. said Officer Stephen Stem told Golden to drop her weapon at least three times.\nStem fired three times, and Golden was hit at least twice, he said.\nShe was transported to a local hospital, where she died.\nThe Hearne Police Department placed Stem on administrative leave pending the inquiry.\n"We\'re very saddened by this. Everybody in the city government is deeply disappointed that this lady was killed," Russ said. "Now, the investigation is out of our hands. It\'s under the Texas Rangers, which is where we want it to be."\nAccording to police, the Texas Rangers have a revolver believed to have been in Golden\'s possession at the time of the shooting.\nCommunity members told CNN affiliate KBTX that Golden, known affectionately as "Ms. Sully," was a sweet woman.\n"Even if she did have a gun, she is in her 90s," Lawanda Cooke told KBTX. "They could have shot in the air to scare her. Maybe she would have dropped it. I don\'t see her shooting anyone."\nThe case will eventually be presented to a grand jury, which is standard procedure when dealing with officer-involved incidents, Russ said.\nIn the meantime, Hearne City Council members will meet Saturday to discuss Stem\'s employment or whether any disciplinary action will be taken.\n"I would expect people to be upset about this, a young police officer shooting a 93-year-old lady," Russ said. "I\'m upset about it. Most of our citizens are upset but at the same time I don\'t believe all the facts have come to the surface yet."\nParents doubt official account of how their son was shot by officer\n\'We called for help and they killed him\'',
            filename=None,
            chunks_pos=None,
            metadata=None,
        ),
    ]

    expected_selections = [
        # exact match
        "A group of citizens are organizing a candlelight vigil. It will be held tonight at 7:00, in front of Golden's home.",
        # normalized match
        "i would expect people to be upset about this",
        # Fuzzy match
        "Residents are questioning why police would shoot Golden who they described as a sweet woman.",
        # Hallucination
        "aliens are watching us.",
    ]
    selected_text = formatter(expected_selections)

    # parse selections
    parsed_selections = grounding.parse_selection(selected_text)
    assert expected_selections == parsed_selections

    # ground selections
    selections = grounding.ground_selections(parsed_selections, docs=docs)
    assert selections == [
        Selection(
            doc_id=0,
            start_pos=886,
            end_pos=1001,
            content="A group of citizens are organizing a candlelight vigil. It will be held tonight at 7:00, in front of Golden's home.",
            metadata={"total_count": 1, "mode": "exact_match"},
        ),
        Selection(
            doc_id=1,
            start_pos=1897,
            end_pos=1941,
            content="I would expect people to be upset about this",
            metadata={"total_count": 1, "mode": "normalized_match"},
        ),
        Selection(
            doc_id=0,
            start_pos=2894,
            end_pos=2993,
            content="Residents are questioning why police would shoot Golden who they described as a sweet, sweet woman.",
            metadata={"total_count": 1, "best_dist": 2, "mode": "fuzzy_match"},
        ),
        Selection(
            doc_id=-1,
            start_pos=-1,
            end_pos=-1,
            content="aliens are watching us.",
            metadata={"total_count": 0, "mode": "hallucination"},
        ),
    ]
