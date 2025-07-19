import pytest

from igcs.datasets.create_gencs_dataset import response_parser

GEMINI_TYPE1 = """```
1. Select content that describes the historical context of piracy during the early 18th century.
    *   Few persons of the present day are aware how extensively piracy prevailed two centuries ago. There was no part of the high seas that was free from the depredation of roving robbers. At times they threatened towns on the coast, and at others they attacked ships on mid-ocean; and they seem to have followed their lawless pursuits at will. When caught, there was little delay in bringing them to trial and securing a conviction; and trivial technicality in forms played no part in reaching results. At times there were multiple executions, and in the community there was no morbid sentimentality shown for the miserable wretches. Not the least of their torture was sitting in the meeting-house on the Sunday before execution and listening to their own funeral sermons, when the minister told them what they might expect in the next world if they got their just dues. On June 30, 1704, six poor victims were hung, on the Boston side of the Charles River bank, for piracy and murder; and there was a great crowd to witness the tragedy. Among the spectators on this occasion was Chief-Justice Sewall, one of the judges of the Admiralty Court which had convicted the pirates, who did not think it beneath his dignity to be present. It was then considered a public duty to invest the scene of execution with as much awe as possible, and it was thought that official station would emphasize this feeling.

2. Select content that details a specific pirate attack, including the names of the people involved and the location of the incident.
    *   On Saturday 19th inst. between the Bahama Bank and Key Sal Bank we were boarded and taken possession of by a small schr. of about 30 tons, having one gun mounted on a pivot and 30 men. She manned us with twelve men, Spaniards, French, Germans and Americans, and carried us towards the Florida coast. Being arrived on the coast nearly opposite to Havana, the privateer went in shore to reconnoitre, and our ship lay off and on. Next morning she returned with two small vessels, a schooner and sloop. We then all four steered over the reef towards the small islands, and on Tuesday afternoon were brought to anchor in a little harbour formed by the Florida isles and the Martyr's Reef, as snug a hole as buccaniers would wish. 
    *   The privateer was, I think, fitted out from this island. The Captain is a Spaniard, a short man with a remarkable good face, that nobody would suspect to belong to such a gang. The Lieutenant is a Frenchman, a creole of St. Domingo, but called himself an Italian. The man they called Davis, who ordered me to be hanged, is the pilot or sailing master, and their boarding officer. He is an American, belongs to New-York, and was the worst man on board. He is a good looking fellow, something perhaps over the middle size, but the most brutal rascal I ever met. There was another American on board, only a common hand, being a drunkard.--Two <DW64>s are all the residue of the gentlemen with whom I had much acquaintance.

3. Select content that explains the legal and societal responses to piracy during the time period discussed.
    *   When caught, there was little delay in bringing them to trial and securing a conviction; and trivial technicality in forms played no part in reaching results. At times there were multiple executions, and in the community there was no morbid sentimentality shown for the miserable wretches. 

4. Select content that provides a personal anecdote related to piracy from the author or a contemporary figure.
    *   The incident to which I have already alluded, occurred in the latter part of March, off Cape Tres Forcas on the Barbary Coast. One afternoon, as we were sailing along at low speed with little wind, two or three leagues from land, we spied two lateen-rigged feluccas, apparently following us, which at first sight attracted but little attention. Captain Roberts soon became suspicious of their movements and watched them closely, as they were gaining on us. We were going hardly more than two or three knots an hour, having little more than steering way, but they spreading much sail were faster. The captain soon gave orders to have an inventory taken of the firearms on board that could be used in case of need, but these were found to be few in number and in poor condition. The cook was ordered to heat as much boiling water as his small galley would allow, to be ready to repel any attempt to board the vessel. There was great excitement on the bark, and we fully expected to be attacked, but fortunately for us 

5. Select content that outlines the impact of piracy on maritime trade routes and the measures taken to combat it.
    *   The neighborhood of Cuba will be troubled waters until our government shall seriously determine to put down this system of piracy. 
``` 
"""

GEMINI_TYPE2 = """```
[
  [
    "One in 10 Britons often feel lonely, and those aged 18-34 are more likely to worry about being isolated than older adults, according to a Mental Health Foundation report.",
    "While 17% of over-55s worry about being alone, 36% of under-35s do.",
    "The elderly, jobless and those who are disabled are most likely to be affected."
  ],
  [
    "Technology can isolate but is also an unrivalled means of connecting people, the poll of 2,256 people concludes.",
    "Nearly a third of young people questioned for the report said they spent too much time communicating with friends and families online when they should see them in person.",
    "Whether this has any genuine biological impact is unclear, but it has been suggested that physical presence is needed for the hormone oxytocin to be released - believed to be the chemical process underpinning the relationship between social contact and healthy hearts.",
    "The young people we work with tell us that talking to hundreds of people on social networks is not like having a real relationship and when they are using these sites they are often alone in their bedrooms"
  ],
  [
    "The report also found gender differences, with more women than men reporting loneliness, and more likely to feel depressed as a result.",
    "It highlighted the fact that the proportion of people living alone, both male and female, had doubled between 1972 and 2008."
  ],
  [
    "Persistent loneliness is bad for people's mental and physical health and can be linked to stress, heavy drinking and poor diet, says the charity.",
    "Chronic elderly isolation was brought to the fore earlier this year with the deaths of a Northamptonshire husband and wife, whose bodies were found in their frozen home in the middle of Britain's coldest winter."
  ],
  [
    "The report cites the example of the parenting website Netmums, which says that because of contacts made online 10,000 women meet face-to-face every month, reducing the sometimes intense sense of isolation new mothers can experience.",
    "“For example, Manhattan in New York has 50% lone households, more than anywhere else in the United States, yet its 'urban village' model sustains social networks because people habitually use alternative meeting places, including cafes and public spaces,” the authors wrote."
  ]
]
```"""

GEMINI_TYPE3 = """- 1. Select content that describes the reaction of the audience to Wilbur Ross's remarks at the Milken Institute Global Conference.
  - "The crowd laughed, Variety reports, and Ross then said, “The thing was, it didn’t cost the president anything to have that entertainment.”"
  - "After the audience at the Milken Institute Global Conference laughed, per Variety, Ross continued: "The thing was, it didn’t cost the president anything to have that entertainment.""
  - "As the crowd laughed, Ross added: “The thing was, it didn’t cost the president anything to have that entertainment.”"
  - "The crowd chuckled at Ross’s remark, according to Variety, and he added: “The thing was, it didn’t cost the president anything to have that entertainment.”"
- 2. Select content that details the financial cost of the missile strike mentioned by Wilbur Ross.
  - "The cost of a Tomahawk missile, per the Navy budget for fiscal year 2017, is $1.355 million. Given that the strike used 59 missiles, that comes out to roughly $79.9 million for just the missiles alone."
  - "The “entertainment” might not have cost Trump anything, but USA Today calculates that the Tomahawk missiles alone cost about $79.9 million. The strike was launched after Syrian president Bashar al-Assad’s forces killed at least 87 people with chemical weapons, and the Syrian military claimed that at least seven people were killed and nine wounded in the U.S. strike."
- 3. Select content that provides information on the setting and timing of President Trump's announcement to President Xi about the Syria strike.
  - "According to Variety, Ross recalled when he found out about the decision to strike Syria at Mar-a-Lago, when President Trump told Chinese President Xi Jinping about the strike."
  - "The way Ross describes it, on April 6, in the middle of President Trump's summit with Chinese President Xi Jinping, "just as dessert was being served, the president explained to Mr. Xi he had something he wanted to tell him, which was the launching of 59 missiles into Syria. It was in lieu of after-dinner entertainment."
  - "Now Commerce Secretary Wilbur Ross has pulled off an impressive feat: combining the Marie Antoinette element of Trump’s remarks and the callousness of Williams’s commentary. During a speech on Monday, Ross recalled the scene at Mar-a-Lago on April 6, when Trump dined with Chinese president Xi Jinping as two navy destroyers launched 59 Tomahawk missiles into Syria."
  - "Speaking at the Milken Institute Global Conference on Monday, Commerce Secretary Wilbur Ross recalled the scene at Mar-a-Lago on April 6, when the summit with Chinese President Xi Jinping was interrupted by the strike on Syria."
- 4. Select content that includes any third-party commentary or reactions on Twitter to Wilbur Ross's remarks.
  - "Perhaps instead of a standup comic or lounge singer? https://t.co/ml9CJLlYOl — Andrea Mitchell (@mitchellreports) May 1, 2017"
  - "This is a most unfortunate quote and headline for Wilbur Ross https://t.co/QtkhPjZqTu — Blake Hounshell (@blakehounshell) May 1, 2017"
  - "[image via screengrab]"
- 5. Select content that explains Wilbur Ross's role and presence during the Syria missile strike as described in the documents.
  - "Ross was in the secured conference room at Mar-a-Lago where Trump and other administration officials huddled during the strike, though he's not necessarily charged with national security interests as head of the Commerce Department." 
  - "After the strike, he noted that the attack took out 20% of the Syrian air force, a number that was later confirmed by the Pentagon." 
"""

GEMINI_TYPE4 = """```
[
  [
    "Select content that details the conversation between Ken Lay and David Sokol regarding the new legislative amendment.",
    "Yesterday, Ken Lay spoke with MidAmerican CEO David Sokol regarding the \nattached new legislative amendment to provide a limited PUHCA exemption for \ncompanies with a high level of financial stability as determined by \nindependent market analysts (i.e. \"A\" rating for Holding Companies or \ninvestment grade rating for utility companies).  This is not a permanent \nexemption as PUHCA would again apply should the company loose its high rating.",
    "MidAmerican's lobbyist reported this conversation to me along with Ken Lay's \nreported commitment that we would review it as soon as we could.  Reportedly, \nKen Lay was non-committal but not negatively disposed."
  ],
  [
    "Select content that outlines the utility strategy involving Senator Bob Kerrey and the specific legislation mentioned.",
    "The utility strategy is to have Senator Bob Kerrey (D-NE) offer this as an \namendment to a \"must pass\" piece of legislation during the waning days of \nthis Congressional session.  This will be separate from the utilities' \nstrategy to enact Private Use legislation (H.R. 4971, S. 2967 which includes \ntax language to exclude water and sewage connection fees from gross income as \ncontributions to capital and language to increase the amount permitted to \npaid into nuclear decommissioning reserve funds primarily  for Commonwealth \nEdison) and to enact stand-alone Reliability legislation (S. 2071 and H.R. \n2944)."
  ],
  [
    "Select content that describes the potential opposition from Congressman Dingell and Congressman Markey to the PUHCA exemption.",
    "It is possible that Congressman Dingell (D-MI) and Congressman Markey (D-MA) \nwould still actively oppose this limited exemption, especially given the fact \nthat Congressman Dingell (who would Chair the Commerce Committee should the \nDemocrats regain control of the House) has recently said that whether or not \nhe is Commerce Committee Chairman next Congress, that comprehensive \nelectricity restructuring legislation including PUHCA repeal is \"at least \nthree years away from happening.\""
  ],
  [
    "Select content that summarizes Enron's historical position on PUHCA repeal and its strategic implications.",
    "Steve and Rick, can we get a temperature gauge next week as to our position \non this amendment?  As a reminder, Enron's position here-to-date has been to \noppose stand-alone PUHCA repeal as this is the one provision utilities like \nSouthern Company really want.  To date, this strategy has worked with the \nMidAmerican Utility Group willing to support our transmission open access \nprovisions in turn for our PUHCA exemption support."
  ],
  [
    "Select content that mentions the follow-up actions planned with Senator Gramm's staff regarding the PUHCA amendment.",
    "Senator Kerrey has already spoken to Senator Gramm (R-TX) about this PUHCA \namendment and Gramm was reportedly  non-committal but not negatively disposed \nto the idea.  I will follow-up with Gramm's staff next week after they've \nreviewed it and convey their views to you.  According to MidAmerican,  the \nconcept to insure stockholder protection along with consumer protection was \nSenator Kerrey's idea."
  ]
]
```"""

GEMINI_TYPE5 = """```
[
  "1. Select content that describes the unique features of the toys mentioned in the document.",
  [
    "Here's a famous managerie, full of wild beasts;\n      See! this lion with wide open jaws,\n    Enough to affright one, and yet I've no doubt,\n      You might venture to play with his claws.",
    "Here's a tiger as tame as a lap-dog, you'll find,\n      And a fox that will not steal the geese:\n    So here you must own the old adage is proved,\n      That wonders are never to cease.",
    "Here's a whole file of soldiers, quite ready for fight,\n      And each of them armed with a gun;\n    You may knock them all down with a feather, and then\n      You may pocket them--every one.",
    "Here's a fine stud of horses, which, strange though it sounds,\n      Live neither on corn nor on hay;\n    A gentleman's carriage, and tilbury, too,\n      For which we've no taxes to pay.",
    "A coachman so plump, and a footman so tall,\n      Who cost not a penny for food;\n    For to tell you the truth, all their insides are filled\n      With a permanent dinner of wood.",
    "Examine this sword, with its handle and sheath,\n      And its blade made of innocent wood;\n    'Twere well if all swords were as harmless as this,\n      And as equally guiltless of blood.",
    "Here's a mill that will go without water or wind,\n      A wonder, you cannot deny;\n    I really can't say whether it will grind corn,\n      But it will be easy to try.",
    "That iron-gray rocking-horse, close at your side,\n      With saddle and bridle complete,\n    Will go without whipping, and, equally strange,\n      Without making use of his feet:",
    "Yet, stranger than that--whatsoever his pace,\n      Whether canter, or gallop, or trot,\n    Though moving at ten miles an hour--he ne'er\n      Advances one inch from the spot.",
    "A full set of bricks is enclosed in this box,\n      (With the mortar we well may dispense,)\n    But with these you may build a magnificent house,\n      Without e'en a farthing's expense.",
    "With these you may raise up a Royal Exchange,\n      In less than five minutes, and then\n    Knock it down, and build up a new Parliament House,\n      In another five minutes,--or ten."
  ],
  "2. Select content that illustrates interactions between the toyman and the children.",
  [
    "\"Pray, what would you like?\" said a Toyman, one day,\n      Addressing a group of young folks,\n    \"I have toys in abundance, and very cheap, too,\n      Though not quite so cheap as my jokes."
  ],
  "3. Select content that details the physical appearance and construction of any toy.",
  [
    "That iron-gray rocking-horse, close at your side,\n      With saddle and bridle complete,"
  ],
  "4. Select content that includes any form of illustration description or reference.",
  [
    "[Illustration: TOY WAREHOUSE.]",
    "[Illustration]",
    "[Illustration]",
    "[Illustration]",
    "[Illustration]",
    "[Illustration]",
    "[Illustration]",
    "[Illustration]"
  ],
  "5. Select content that provides information on the publication details of the document.",
  [
    "                          THE\n                          WONDERS\n                           OF A\n                            TOY\n                           SHOP.",
    "                          New-York:\n                       J. Q. PREBLE.",
    "                     J. W. ORR NEW YORK."
  ]
]
```"""

GEMINI_TYPE6 = """- **1. Select content that describes the proximity of each hotel to major transportation routes or highways.** 
  - "Hotel is well situated only a few minutes drive from the M50 motorway/ Albecete motorway access, and provides free parking for guests." (Document #2)
  - "The hotel is 2 mins off the main motorway between Madrid and Valencia." (Document #12)
  - "It is 15 minutes drive from Madrid and the directions can be confusing the first time." (Document #13)
  - "Hotel is near industrial area." (Document #6)
  - "It's located in the middle of nowhere, very industrial area." (Document #7)
  - "The hotel is way out from city center." (Document #9)
  - "This hotel is good on location, nearly to Rivas Centro which is there the mini mall situated in that area." (Document #10)
  - "Next to 8 lane highway under flight path with nothing in walking distance" (Document #11) 
- **2. Select content that details the availability and quality of breakfast options at each hotel.**
  - "poor breakfast" (Document #1)
  - "Compimentary breakfast was very good, considering the price paid for the double room." (Document #2)
  - "Breakfast not bad." (Document #3)
  - "Breakfast is nice and simple.. there aren't too much were we can choose from, The main issue i found during the breakfast, was the existence of the machines in one single side, so, there was a long line to use them.. (waste of time)..." (Document #4)
  - "no decent breakfast" (Document #5)
  - "We picked hotels online that listed accommodations for parking, WiFi internet connection, breakfast." (Document #6)
  - "there's free breakfast" (Document #8)
  - "The only view is from the breakfast area on the main floor" (Document #9)
  - "They include a free continental breakfast that consists of bread & rolls, 3 types of cereal, juices, coffee, milk, hot chocolate, cakes/donuts, tinned fruit." (Document #10)
  - "Breakfast is complimentary, what that means is it is cheap and nasty. If you like dry bread, rubber ham and cheese you will luv it." (Document #11)
  - "Breakfast was good" (Document #12)
  - "very nice breakfast is included in the rate" (Document #13)
- **3. Select content that mentions any specific amenities or services that are lacking in each hotel.**
  - "no fridge or safe box in the room" (Document #1)
  - "no fridge or safe in the room." (Document #4)
  - "no room service, no restaurants inside the hotel, no decent breakfast, no complementary water in the room.." (Document #5)
  - "Charge for basic TV channels." (Document #6)
  - "First off there is no shuttle service from the airport so your paying at least 20+ Euro there and back each way." (Document #7)
  - "No pool, bar is smaller than my kitchen." (Document #11)
- **4. Select content that evaluates the cleanliness and maintenance status of each hotel.**
  - "The bad are ok: you have a good sleep." (Document #1) 
  - "The hotel rooms are clean, well furnished and ideal for a overnight stay." (Document #2)
  - "room was clean." (Document #3)
  - "Rooms have nice accomodation and everything was cleaned every day... sheets, pilows, etc.." (Document #4)
  - "Looks clean." (Document #6)
  - "The place is very clean--the cleaning staff works until 2000 every day. The rooms are modern, you don't have to pay for tv, there's free breakfast, the staff is amazingly helpful, and there's a nice bar." (Document #8)
  - "It is run down, really showing the wear and tear." (Document #9)
  - "Room are comfy and a good enough size. Sheets changed every 2 day or whenever you want." (Document #10)
  - "The hotel is exactly what you expect from a HI Express, clean, functional but not luxuirious." (Document #12)
  - "Rooms were very clean and staff very helpful." (Document #13)
- **5. Select content that provides information about the surrounding area of each hotel, such as nearby restaurants or shopping centers.**
  - "The hotel is in a commercial area (many clothing stores around), but also industry." (Document #1)
  - "The hotel itself is situated on a development which contains a shopping centre, McDonalds and several warehouse/business premises." (Document #2)
  - "The shopping centre is only 5 minutes walk from the hotel and provides numerous retail outlets and a large supermarket. Likewise, McDonalds is just 2 minutes walk away." (Document #2)
  - "we can find restaurantes nearby, even a mcdonals and a mall." (Document #4)
  - "For dinner, there are no nearby restaurants save a steakhouse. The steakhouse is good, not great. The portions are small and the prices hefty. If you need to eat, order from the Chinese take-out place; wholly moley, it was surprisingly good." (Document #8)
  - "There is a small mall a couple blocks away, but in driving around on our own found a major mall about 2 km away that had restaurants open late." (Document #9)
  - "This hotel is good on location, nearly to Rivas Centro which is there the mini mall situated in that area." (Document #10)
  - "There are no shops or amenities you can walk to. You are basically stuck next to a building site and wast areas." (Document #11)
  - "Nearby was a McDonalds and a shopping centre (2 mins drive, 10 mins walk - which included about 5 different chains of restuarant and a supermarket, together with free parking underneath)." (Document #12) 
"""

GEMINI_TYPE7 = """```
[
  {
    "instruction": "Select content that describes the historical significance of The Roebuck in relation to Epping Forest.",
    "content": [
      "THE ROEBUCK GARDENS AND GROUNDS have always been historically associated\nwith the adjacent Forest, and the quaint old edifice has been referred to\nchiefly as the Foresters’ and Keeper’s Home for more than two centuries,\nso much so, that it was under the consideration of the late proprietors,\nMessrs. Green Brothers, at the suggestion of their neighbours and\nvisitors to name it THE LORD WARDEN’S ROEBUCK HOTEL!"
    ]
  },
  {
    "instruction": "Select content that outlines the impact of railway development on the popularity of Epping Forest.",
    "content": [
      "The popularity of this place was enhanced considerably by the formation\nof the Loughton, Woodford, and Ongar branch of the Eastern Counties\nRailway, although, prior to that, the prejudices against Essex scenery\nhad kept many persons, who now wander about its sunny <DW72>s with unmixed\ndelight, from seeking air and exercise North-east of the Metropolis"
    ]
  },
  {
    "instruction": "Select content that details the specific scenic features and geographical location of The Roebuck.",
    "content": [
      "The situation (on the brow of a lofty hill), with two deep valleys on\neither side of it, watered by the rivers Lea and Roding, is scarcely to\nbe rivalled, as to scenery, even by the far-famed contiguous eminence of\nHigh Beech.",
      "In the extreme distance is the ancient town of Epping, from which the\nForest takes its name, and “ye wodes of Waltham,” referred to in\n“Doomsday Book,” are on the opposite heights.",
      "To the North-west is the\ncave of the renowned Turpin; and this haunt of the Essex freebooter may\nbe seen from hence, and easily reached by descending a ravine and\nclimbing the high hill beyond it."
    ]
  },
  {
    "instruction": "Select content that mentions any literary figures associated with Epping Forest and their contributions or experiences there.",
    "content": [
      "To the lovers of poetry this place will be interesting, inasmuch as at\nFair Mead Bottom the author of the “Pleasures of Hope” lived in sedation,\nbut so great was his love for the retreat we are now describing, that he\n(Thomas Campbell) half cut a way to it with a knife, and although this\nvista was relinquished through his death, it was finished by a gentleman\nof the same name, who resided at the Hotel for years, he remarking, with\nemphasis, that “A Campbell began it, and a Campbell completed it.”",
      "Another great author, the late Charles Dickens, no later than about seven\nyears since, in a conversation that he had with the proprietors, Messrs.\nGreen Brothers, stated that it have him great pleasure to visit this\nhouse, inasmuch as he had always considered it as the central rendezvous\nfor all Foresters from time immemorial."
    ]
  },
  {
    "instruction": "Select content that provides information about the facilities and services available at The Roebuck as of the document's publication date.",
    "content": [
      "This antique edifice, the HOTEL, which is\ndetached from the high road to Cambridge about a furlong, is approached\nby a semi-circular carriage way which diverges from the above road on the\nsummit of Buckhurst Hill, re-entering the same further down towards\nLoughton.",
      "It is provided with an ample bar and airy and well ventilated\napartments, overlooking prospects principally of immense tracts of\nforest, relieved by corn fields and undulating meads.",
      "Adjoining the\nhotel a Ball or Banquetting Hall has been erected, capable of dining 500\npersons; indeed the proprietors found it necessary, to meet the\ncontinually increasing demands for large Annual Dinners, Masonic\nBanquets, Fetes, &c.",
      "When it is borne in mind that these Grounds cover over 22 acres, and that\nthe greater part of this area is laid out in Gardens, Terraces, Bowling\nGreens, &c., with a profusion of Flowers, some idea may be formed of the\nwhole, but it must be visited and inspected, since no description can\npossibly convey an adequate idea of the place.",
      "There is every accommodation for Horses, Carriages, &c., and the\nBuckhurst Hill Railway Station is little more than ten minutes’ walk from\nthis ancient hostelry."
    ]
  }
]
```"""

CLAUDE_TYPE1 = """1. Select code that initializes the `relative` list in the `ServerPlayerPositionRotationPacket` class.
   - ```java
     private @NonNull List<PositionElement> relative;
     ```
   - ```java
     public ServerPlayerPositionRotationPacket(double x, double y, double z, float yaw, float pitch, int teleportId, boolean dismountVehicle, PositionElement... relative) {
         this(x, y, z, yaw, pitch, teleportId, dismountVehicle, Arrays.asList(relative != null ? relative : new PositionElement[0]));
     }
     ```

2. Select code that handles the serialization of position and rotation data in the `ServerPlayerPositionRotationPacket` class.
   - ```java
     @Override
     public void read(NetInput in) throws IOException {
         this.x = in.readDouble();
         this.y = in.readDouble();
         this.z = in.readDouble();
         this.yaw = in.readFloat();
         this.pitch = in.readFloat();
     
         this.relative = new ArrayList<>();
         int flags = in.readUnsignedByte();
         for(PositionElement element : PositionElement.values()) {
             int bit = 1 << MagicValues.value(Integer.class, element);
             if((flags & bit) == bit) {
                 this.relative.add(element);
             }
         }
     
         this.teleportId = in.readVarInt();
         this.dismountVehicle = in.readBoolean();
     }
     ```
   - ```java
     @Override
     public void write(NetOutput out) throws IOException {
         out.writeDouble(this.x);
         out.writeDouble(this.y);
         out.writeDouble(this.z);
         out.writeFloat(this.yaw);
         out.writeFloat(this.pitch);
     
         int flags = 0;
         for(PositionElement element : this.relative) {
             flags |= 1 << MagicValues.value(Integer.class, element);
         }
     
         out.writeByte(flags);
     
         out.writeVarInt(this.teleportId);
         out.writeBoolean(this.dismountVehicle);
     }
     ```

3. Select code that reads and writes the `health`, `food`, and `saturation` attributes in the `ServerPlayerHealthPacket` class.
   - ```java
     @Override
     public void read(NetInput in) throws IOException {
         this.health = in.readFloat();
         this.food = in.readVarInt();
         this.saturation = in.readFloat();
     }
     ```
   - ```java
     @Override
     public void write(NetOutput out) throws IOException {
         out.writeFloat(this.health);
         out.writeVarInt(this.food);
         out.writeFloat(this.saturation);
     }
     ```

4. Select code that calculates the flags for position elements in the `write` method of the `ServerPlayerPositionRotationPacket` class.
   - ```java
     int flags = 0;
     for(PositionElement element : this.relative) {
         flags |= 1 << MagicValues.value(Integer.class, element);
     }
     ```

5. Select code that demonstrates the use of the `MagicValues` class in the `ServerPlayerPositionRotationPacket` class.
   - ```java
     int bit = 1 << MagicValues.value(Integer.class, element);
     ```
   - ```java
     flags |= 1 << MagicValues.value(Integer.class, element);
     ```
"""

GPT4_TYPE1 = """Here are the code selections based on the given instructions:

```json
[
    [
        "Select code that imports modules from '@angular/core'.",
        [
            "import { Inject, Injectable, NgModule, createPlatformFactory } from '@angular/core';"
        ]
    ],
    [
        "Select code that defines a class which extends TestComponentRenderer.",
        [
            "class DOMTestComponentRenderer extends TestComponentRenderer {",
            "    constructor(_doc /** TODO #9100 */) {",
            "        super();",
            "        this._doc = _doc; /** TODO #9100 */",
            "    }",
            "    insertRootElement(rootElId) {",
            "        const rootEl = ɵgetDOM().firstChild(ɵgetDOM().content(ɵgetDOM().createTemplate(`<div id=\"${rootElId}\"></div>`)));",
            "        // TODO(juliemr): can/should this be optional?",
            "        const oldRoots = ɵgetDOM().querySelectorAll(this._doc, '[id^=root]');",
            "        for (let i = 0; i < oldRoots.length; i++) {",
            "            ɵgetDOM().remove(oldRoots[i]);",
            "        }",
            "        ɵgetDOM().appendChild(this._doc.body, rootEl);",
            "    }",
            "}"
        ]
    ],
    [
        "Select code that involves manipulation of DOM elements within a method.",
        [
            "const rootEl = ɵgetDOM().firstChild(ɵgetDOM().content(ɵgetDOM().createTemplate(`<div id=\"${rootElId}\"></div>`)));",
            "const oldRoots = ɵgetDOM().querySelectorAll(this._doc, '[id^=root]');",
            "for (let i = 0; i < oldRoots.length; i++) {",
            "    ɵgetDOM().remove(oldRoots[i]);",
            "}",
            "ɵgetDOM().appendChild(this._doc.body, rootEl);"
        ]
    ],
    [
        "Select code that details the decorators used for the BrowserDynamicTestingModule class.",
        [
            "BrowserDynamicTestingModule.decorators = [",
            "    { type: NgModule, args: [{",
            "                exports: [BrowserTestingModule],",
            "                providers: [",
            "                    { provide: TestComponentRenderer, useClass: DOMTestComponentRenderer },",
            "                ]",
            "            },] },",
            "];"
        ]
    ],
    [
        "Select code that exports the platformBrowserDynamicTesting function.",
        [
            "export { platformBrowserDynamicTesting, BrowserDynamicTestingModule, DOMTestComponentRenderer as ɵDOMTestComponentRenderer };"
        ]
    ]
]
```"""


def assert_selection(selection: str, expected_last_selections: str, expected_last_items: list[str]):
    res = response_parser.parse_multi_selections(selection)
    assert len(res) == 5
    assert list(res)[4] == expected_last_selections
    assert list(res.values())[4] == expected_last_items


def test_parse_type1():
    assert_selection(
        GEMINI_TYPE1,
        "Select content that outlines the impact of piracy on maritime trade routes "
        "and the measures taken to combat it.",
        [
            "The neighborhood of Cuba will be troubled waters until our government shall seriously determine to put down this system of piracy"
        ],
    )


def test_parse_type2():
    assert_selection(
        GEMINI_TYPE2,
        "4",
        [
            "The report cites the example of the parenting website Netmums, which says that because of contacts made online 10,000 women meet face-to-face every month, reducing the sometimes intense sense of isolation new mothers can experience.",
            "“For example, Manhattan in New York has 50% lone households, more than anywhere else in the United States, yet its 'urban village' model sustains social networks because people habitually use alternative meeting places, including cafes and public spaces,” the authors wrote.",
        ],
    )


def test_parse_type3():
    assert_selection(
        GEMINI_TYPE3,
        "Select content that explains Wilbur Ross's role and presence during the "
        "Syria missile strike as described in the documents.",
        [
            "Ross was in the secured conference room at Mar-a-Lago where Trump and other administration officials huddled during the strike, though he's not necessarily charged with national security interests as head of the Commerce Department",
            "After the strike, he noted that the attack took out 20% of the Syrian air force, a number that was later confirmed by the Pentagon",
        ],
    )


def test_parse_type4():
    assert_selection(
        GEMINI_TYPE4,
        "Select content that mentions the follow-up actions planned with Senator Gramm's staff regarding the PUHCA amendment.",
        [
            "Senator Kerrey has already spoken to Senator Gramm (R-TX) about this PUHCA \namendment and Gramm was reportedly  "
            "non-committal but not negatively disposed \nto the idea.  I will follow-up with Gramm's staff next week after they've "
            "\nreviewed it and convey their views to you.  According to MidAmerican,  the \nconcept to insure stockholder protection "
            "along with consumer protection was \nSenator Kerrey's idea."
        ],
    )


def test_parse_type5():
    assert_selection(
        GEMINI_TYPE5,
        "5. Select content that provides information on the publication details of the document.",
        [
            "                          THE\n                          "
            "WONDERS\n                           "
            "OF A\n                            TOY\n                           SHOP.",
            "                          New-York:\n                       J. Q. PREBLE.",
            "                     J. W. ORR NEW YORK.",
        ],
    )


def test_parse_type6():
    assert_selection(
        GEMINI_TYPE6,
        "Select content that provides information about the surrounding area of each hotel, such as nearby restaurants or shopping centers.**",
        [
            '"The hotel is in a commercial area (many clothing stores around), but also '
            'industry." (Document #1)',
            '"The hotel itself is situated on a development which contains a shopping '
            'centre, McDonalds and several warehouse/business premises." (Document #2)',
            '"The shopping centre is only 5 minutes walk from the hotel and provides '
            "numerous retail outlets and a large supermarket. Likewise, McDonalds is just "
            '2 minutes walk away." (Document #2)',
            '"we can find restaurantes nearby, even a mcdonals and a mall." (Document #4)',
            '"For dinner, there are no nearby restaurants save a steakhouse. The '
            "steakhouse is good, not great. The portions are small and the prices hefty. "
            "If you need to eat, order from the Chinese take-out place; wholly moley, it "
            'was surprisingly good." (Document #8)',
            '"There is a small mall a couple blocks away, but in driving around on our '
            'own found a major mall about 2 km away that had restaurants open late." '
            "(Document #9)",
            '"This hotel is good on location, nearly to Rivas Centro which is there the '
            'mini mall situated in that area." (Document #10)',
            '"There are no shops or amenities you can walk to. You are basically stuck '
            'next to a building site and wast areas." (Document #11)',
            '"Nearby was a McDonalds and a shopping centre (2 mins drive, 10 mins walk - '
            "which included about 5 different chains of restuarant and a supermarket, "
            'together with free parking underneath)." (Document #12)',
        ],
    )


def test_parse_gemini_type7():
    assert_selection(
        GEMINI_TYPE7,
        "Select content that provides information about the facilities and services available at The Roebuck as of the document's publication date.",
        [
            "This antique edifice, the HOTEL, which is\ndetached from the high road to Cambridge about a furlong, is approached\n"
            "by a semi-circular carriage way which diverges from the above road on the\n"
            "summit of Buckhurst Hill, re-entering the same further down towards\n"
            "Loughton.",
            "It is provided with an ample bar and airy and well ventilated\n"
            "apartments, overlooking prospects principally of immense tracts of\n"
            "forest, relieved by corn fields and undulating meads.",
            "Adjoining the\nhotel a Ball or Banquetting Hall has been erected, capable of dining 500\n"
            "persons; indeed the proprietors found it necessary, to meet the\n"
            "continually increasing demands for large Annual Dinners, Masonic\n"
            "Banquets, Fetes, &c.",
            "When it is borne in mind that these Grounds cover over 22 acres, and that\n"
            "the greater part of this area is laid out in Gardens, Terraces, Bowling\n"
            "Greens, &c., with a profusion of Flowers, some idea may be formed of the\n"
            "whole, but it must be visited and inspected, since no description can\n"
            "possibly convey an adequate idea of the place.",
            "There is every accommodation for Horses, Carriages, &c., and the\n"
            "Buckhurst Hill Railway Station is little more than ten minutes’ walk from\n"
            "this ancient hostelry.",
        ],
    )


def test_parse_claude_type1():
    assert_selection(
        CLAUDE_TYPE1,
        "Select code that demonstrates the use of the `MagicValues` class in the `ServerPlayerPositionRotationPacket` class.",
        [
            "int bit = 1 << MagicValues.value(Integer.class, element);",
            "flags |= 1 << MagicValues.value(Integer.class, element);",
        ],
    )


def test_parse_gpt4_type1():
    assert_selection(
        GPT4_TYPE1,
        "Select code that exports the platformBrowserDynamicTesting function.",
        [
            "export { platformBrowserDynamicTesting, BrowserDynamicTestingModule, "
            "DOMTestComponentRenderer as ɵDOMTestComponentRenderer };"
        ],
    )
