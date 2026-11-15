from __future__ import annotations

from dataclasses import dataclass
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class ComplaintClassifier:
    name: str
    description: str
    section_numbers: tuple[int, ...]
    keywords: tuple[str, ...]
    examples: tuple[str, ...]


# ---------------------------------------------------------------------------
# TRAINING DATA
# ---------------------------------------------------------------------------

CLASSIFIERS: tuple[ComplaintClassifier, ...] = (
    ComplaintClassifier(
        "Theft",
        "Taking property without permission",
        (303, 304, 309, 316, 324, 329),
        (
            "stole", "stolen", "theft", "taken", "took", "missing",
            "wallet", "mobile", "phone", "bag", "purse", "bicycle",
            "laptop", "watch", "jewelry", "jewellery", "chain",
        ),
        (
            "My mobile phone was stolen from my bag.",
            "Someone stole my bicycle from outside my house.",
            "My laptop was stolen from my room.",
            "My wallet was stolen in the market.",
            "Someone took my phone without my permission.",
            "My bag was stolen while travelling.",
            "My gold chain was stolen.",
            "Someone stole my watch from my house.",
            "I left my phone on the bus and when I came back it was gone.",
            "My phone disappeared from my desk while I was away.",
            "I noticed that my wallet was missing after visiting the market.",
            "Someone took my bicycle while it was parked outside.",
            "My laptop disappeared from my room.",
            "I think someone removed my phone from my backpack.",
            "My purse went missing while I was shopping.",
            "Someone took my belongings while I was asleep.",
            "I came home and found that my watch was missing.",
            "My jewelry was taken from my room.",
            "Someone removed my phone from my bag without asking.",
            "My headphones were taken from my desk.",
            "I cannot find my wallet and believe somebody took it.",
            "Someone took my bag while I was travelling.",
            "My bicycle was there in the morning but was gone later.",
            "My phone was taken while I was at the restaurant.",
            "Someone took cash from my bag.",
            "My laptop was taken when I left my room briefly.",
            "I discovered that my watch had disappeared.",
            "Someone took my purse from the chair.",
            "My belongings were missing when I returned home.",
            "Someone took my phone from my table.",
            "My backpack was taken from the bus.",
            "I believe somebody took my wallet from my pocket.",
            "My chain disappeared while I was outside.",
            "Someone took my camera from my room.",
            "My bicycle was removed from outside my house.",
            "I returned to my seat and my phone was gone.",
            "Someone took my bag without my consent.",
            "My valuables disappeared from my room.",
            "I found out later that someone had taken my laptop.",
            "My phone was missing after I left it unattended.",
        ),
    ),

    ComplaintClassifier(
        "Cyber Fraud",
        "Online deception or payment fraud",
        (318, 319, 336, 351),
        (
            "otp", "bank", "banking", "link", "online transaction",
            "payment request", "fraudster", "upi", "unauthorized",
            "transaction", "account", "verification", "payment",
            "transfer",
        ),
        (
            "Someone cheated me through an online transaction.",
            "I received a fake banking link and lost money.",
            "A fraudster tricked me into sharing my OTP.",
            "Someone made an unauthorized online transaction.",
            "My bank account was accessed by a fraudster.",
            "I was cheated through a fake online payment.",
            "Someone sent me a fake payment request.",
            "A person pretending to be a bank employee cheated me.",
            "Someone called claiming to be from my bank and took my money.",
            "I clicked a link that looked like my bank's website and lost money.",
            "A person convinced me to share an OTP and money was taken from my account.",
            "Someone made a UPI transaction without my permission.",
            "I received a message saying my bank account would be blocked unless I paid.",
            "A fake customer-care representative tricked me into making a payment.",
            "Someone impersonated a bank employee during a phone call.",
            "Money was transferred from my account without my approval.",
            "I was tricked by a fake payment link.",
            "Someone asked me to verify my account and then money disappeared.",
            "A person sent me a fraudulent UPI payment request.",
            "I received a fake banking message and followed its instructions.",
            "Someone accessed my banking account after contacting me online.",
            "A fraudster convinced me that I needed to share my OTP.",
            "I was tricked into transferring money through an online payment app.",
            "Someone used my account to make an unauthorized transaction.",
            "I received a suspicious link pretending to be from my bank.",
            "A fake bank representative asked me for account details.",
            "Someone convinced me to scan a QR code and money was deducted.",
            "I was contacted by someone pretending to be customer support.",
            "A person tricked me into approving a UPI transaction.",
            "My account was charged after I responded to a fraudulent message.",
            "Someone sent me a fake refund link and I lost money.",
            "I was told I had to pay a fee to receive a refund, and money was taken.",
            "A fake delivery message led me to a payment page.",
            "Someone pretended to help me with my bank account and cheated me.",
            "An unauthorized online payment was made from my account.",
            "I entered my banking information on a fake website.",
            "Someone deceived me into making an online transfer.",
            "I received a fraudulent payment request from an unknown person.",
            "A person pretending to work for my bank obtained my banking details.",
            "I lost money after following instructions from a fake bank representative.",
        ),
    ),

    ComplaintClassifier(
        "Cyberbullying",
        "Repeated hostile or threatening online conduct",
        (78, 79, 351, 352, 356),
        (
            "bullying", "bullied", "social media", "targeting",
            "embarrassing comments", "hateful messages", "mocking",
            "humiliating", "insulting", "ridiculing",
        ),
        (
            "Someone is repeatedly sending insulting messages online.",
            "My classmate is bullying me on social media.",
            "People are posting embarrassing comments about me online.",
            "Someone is threatening me through social media messages.",
            "A person keeps making fun of me online.",
            "Someone is spreading hateful messages about me.",
            "I am being bullied through online messages.",
            "Someone keeps targeting me on social media.",
            "A group of people repeatedly mocks me online.",
            "Someone keeps posting insulting comments under my photos.",
            "My classmates are repeatedly making fun of me online.",
            "Someone created posts specifically to humiliate me.",
            "I keep receiving cruel messages from the same person.",
            "People are repeatedly targeting me in a group chat.",
            "Someone is constantly ridiculing me online.",
            "A person keeps making embarrassing posts about me.",
            "Several people are joining in to mock me online.",
            "Someone repeatedly tags me in insulting posts.",
            "I am being targeted with hateful comments on social media.",
            "A person keeps sharing humiliating comments about me.",
            "Someone is repeatedly making fun of me in an online group.",
            "My classmates keep posting jokes about me to embarrass me.",
            "Someone keeps sending me threatening messages online.",
            "People repeatedly post insulting things about me.",
            "A person has been targeting me through social media for weeks.",
            "Someone keeps trying to embarrass me in public online posts.",
            "I am repeatedly being mocked in a group chat.",
            "A person keeps posting mean comments about me.",
            "Someone repeatedly sends messages designed to humiliate me.",
            "People are targeting me with insulting social media posts.",
            "Someone has been bullying me through messages every day.",
            "A group chat is being used to repeatedly make fun of me.",
            "Someone keeps reposting embarrassing things about me.",
            "A person repeatedly comments insulting things on my posts.",
            "I am being targeted by classmates online.",
            "Someone keeps humiliating me through social media.",
            "I receive repeated hateful messages from another person.",
            "Someone repeatedly mocks me whenever I post something.",
            "People keep attacking me with insulting comments online.",
            "A person has repeatedly targeted me with embarrassing posts.",
        ),
    ),

    ComplaintClassifier(
        "Identity Theft",
        "Impersonation or misuse of identity details",
        (319, 336, 318),
        (
            "identity", "aadhaar", "fake account", "fake profile",
            "fake instagram account", "fake social media account",
            "account using my name", "profile using my name",
            "pretending to be me", "impersonating me",
            "personal information", "impersonating", "impersonation",
            "identity details",
        ),
        (
            "Someone is using my identity without permission.",
            "A person created a fake account using my name.",
            "Someone is using my personal information.",
            "My identity details were stolen online.",
            "Someone used my Aadhaar details without permission.",
            "A fake profile has been created using my identity.",
            "Someone is pretending to be me online.",
            "My personal details were used by another person.",
            "Someone created an account pretending to be me.",
            "A person is using my name on social media.",
            "Someone made a fake profile with my photographs.",
            "A stranger is pretending to be me online.",
            "My personal information was used to create an account.",
            "Someone is impersonating me on social media.",
            "A fake account has been created in my name.",
            "Someone used my identity documents without permission.",
            "A person is pretending to represent me online.",
            "Someone has copied my identity and created a profile.",
            "My Aadhaar information was used by another person.",
            "Someone is using my details to contact other people.",
            "A fake social media account is pretending to be me.",
            "Someone created an account using my name and information.",
            "A person is impersonating me and messaging my friends.",
            "Someone has used my personal details without asking me.",
            "My identity information appeared on an account I did not create.",
            "Someone is pretending to be me to communicate with others.",
            "A fake profile contains my name and photographs.",
            "Someone used my identification details for an account.",
            "A person created a profile using my identity.",
            "Someone is falsely representing themselves as me online.",
            "My personal information was used without my consent.",
            "Someone has copied my profile and is pretending to be me.",
            "A stranger is using my name and photographs online.",
            "Someone created a fake account using details belonging to me.",
            "My identity was used to contact people I know.",
            "Someone is impersonating me through a social media profile.",
            "A person is using my Aadhaar details without my permission.",
            "Someone has created an online account in my name.",
            "My personal details were used by somebody else.",
            "Someone is pretending to be me and communicating with others.",
        ),
    ),

    ComplaintClassifier(
        "Harassment",
        "Repeated unwanted contact, following, or hostile conduct",
        (74, 75, 77, 78, 79, 85, 351),
        (
            "repeatedly", "keeps", "contacting", "unwanted messages",
            "following", "harassed", "bothering", "calling",
            "continuously", "stop contacting",
        ),
        (
            "Someone keeps disturbing and threatening me.",
            "I am being repeatedly harassed by a person.",
            "A person keeps contacting me even after I asked them to stop.",
            "Someone is continuously bothering me.",
            "I am receiving repeated unwanted messages.",
            "A person is following and troubling me.",
            "Someone keeps calling me and disturbing me.",
            "I am being threatened repeatedly.",
            "Someone keeps contacting me despite my requests to stop.",
            "A person repeatedly calls me throughout the day.",
            "Someone follows me whenever I leave school.",
            "I have asked this person to stop contacting me but they continue.",
            "Someone keeps sending unwanted messages to my phone.",
            "A person repeatedly tries to contact me.",
            "Someone keeps bothering me even after I told them to stop.",
            "I receive calls from the same person again and again.",
            "Someone repeatedly waits for me outside.",
            "A person keeps following me when I go home.",
            "Someone continues to contact me despite being told not to.",
            "I am receiving constant unwanted calls.",
            "Someone keeps approaching me even though I avoid them.",
            "A person repeatedly sends messages that I do not want.",
            "Someone has been bothering me for several days.",
            "I told this person to leave me alone but they continue.",
            "Someone repeatedly tries to get my attention against my wishes.",
            "A person keeps coming near me despite my requests.",
            "I am being continuously disturbed by someone.",
            "Someone keeps calling me from different numbers.",
            "A person repeatedly contacts my family about me.",
            "Someone is following me and making me uncomfortable.",
            "I receive unwanted messages from the same person every day.",
            "Someone keeps trying to contact me after I blocked them.",
            "A person repeatedly comes to places where I am.",
            "Someone continues bothering me even after I clearly said no.",
            "I am repeatedly receiving unwanted phone calls.",
            "Someone keeps approaching me and refusing to leave me alone.",
            "A person has been repeatedly disturbing me.",
            "Someone follows me whenever I travel home.",
            "I have repeatedly asked someone not to contact me.",
            "Someone keeps trying to communicate with me despite my objections.",
        ),
    ),

    ComplaintClassifier(
        "Physical Assault",
        "Physical attack, injury, restraint, or use of force",
        (114, 115, 116, 117, 118, 119, 120, 121, 122, 124, 125, 126, 127, 129, 130),
        (
            "punched", "punched me", "slapped", "hit me", "beat me", "kicked",
            "injured", "fracture", "broken bone", "assaulted", "attacked",
            "knife", "weapon", "restrained", "confined", "locked me",
        ),
        (
            "A person punched me and injured me.",
            "Someone hit me and caused an injury.",
            "I was assaulted and suffered a fracture.",
            "A person attacked me with a weapon.",
            "Someone slapped and kicked me.",
            "I was physically attacked by another person.",
            "Someone restrained me and would not let me leave.",
            "A person locked me inside a room.",
            "Someone injured me during an argument.",
            "A person assaulted me and caused serious injury.",
        ),
    ),

    ComplaintClassifier(
        "Financial Scam",
        "Deceptive offers or schemes used to obtain money",
        (308, 316, 318, 319),
        (
            "investment", "loan", "job", "prize", "large return",
            "financial offer", "scheme", "profit", "returns",
            "reward", "deposit",
        ),
        (
            "I lost money in a fake investment scheme.",
            "Someone promised me a job and took my money.",
            "I was cheated by a fake loan company.",
            "A person asked for money promising a large return.",
            "I lost money in an online investment scam.",
            "Someone took money from me promising a fake prize.",
            "I paid money to a fraudulent company.",
            "I was cheated by a fake financial offer.",
            "Someone promised huge profits if I invested money.",
            "I paid a fee for a job that did not exist.",
            "A person promised me a prize after I paid money.",
            "Someone convinced me to invest in a fake company.",
            "I was promised guaranteed returns on my investment.",
            "A fake loan provider took money from me.",
            "Someone offered me a job and demanded an upfront payment.",
            "I deposited money into a scheme that turned out to be fake.",
            "A person promised to double my money.",
            "I was tricked into paying for a fake lottery prize.",
            "Someone asked me for money in exchange for a promised reward.",
            "I lost money after joining a fraudulent investment scheme.",
            "A company offered me a fake financial opportunity.",
            "Someone promised high returns and disappeared after receiving my money.",
            "I paid money to secure a job that was never provided.",
            "A person offered me a fake loan and collected fees.",
            "I was promised a large profit if I gave someone money.",
            "Someone convinced me that an investment was guaranteed.",
            "I paid an advance for a job that did not exist.",
            "A fake investment adviser convinced me to transfer money.",
            "Someone collected money by promising a financial reward.",
            "I lost money through a fraudulent business opportunity.",
            "A person promised to arrange a loan after I paid a fee.",
            "I was cheated by someone offering unrealistic investment returns.",
            "Someone asked for a deposit before giving me a promised job.",
            "I paid money for a prize that I never received.",
            "A fake company convinced me to make an investment.",
            "Someone promised me easy profits in exchange for money.",
            "I lost money after trusting a fraudulent financial offer.",
            "A person took my money after promising a large return.",
            "Someone offered me a fake employment opportunity and demanded payment.",
            "I paid money to participate in a fraudulent financial scheme.",
        ),
    ),

    ComplaintClassifier(
        "Online Abuse",
        "Abusive posts, insults, or reputation-harming claims online",
        (79, 351, 352, 356),
        (
            "abusive", "offensive", "abusing", "harmful content",
            "posted", "uploaded", "comments", "insult", "insulting",
            "false claims", "reputation",
        ),
        (
            "Someone posted abusive content about me online.",
            "A person is using offensive language against me online.",
            "Someone uploaded abusive content about me.",
            "I am receiving abusive comments on social media.",
            "Someone is posting harmful content about me.",
            "A person is using offensive posts against me.",
            "Someone is abusing me through social media.",
            "I am facing abusive behaviour online.",
            "Someone posted insulting comments about me.",
            "A person uploaded offensive material about me.",
            "Someone made harmful claims about me online.",
            "A person is posting insulting content on my profile.",
            "Someone uploaded a post that damages my reputation.",
            "A person keeps writing offensive comments about me.",
            "Someone posted false and harmful statements about me.",
            "I am being insulted through online posts.",
            "Someone is using social media to spread offensive content about me.",
            "A person uploaded embarrassing material about me.",
            "Someone posted abusive comments under my photographs.",
            "A person is making offensive statements about me online.",
            "Someone shared harmful content about me publicly.",
            "A person keeps posting insulting things about me.",
            "Someone uploaded offensive messages about me.",
            "I am receiving abusive comments on my social media account.",
            "Someone posted claims about me that are damaging my reputation.",
            "A person is using an online platform to insult me.",
            "Someone shared an abusive post about me.",
            "A person uploaded content intended to humiliate me.",
            "Someone is posting offensive material using my name.",
            "I found several abusive comments directed at me online.",
            "Someone published harmful allegations about me online.",
            "A person keeps uploading insulting content about me.",
            "Someone posted offensive statements about me in a group.",
            "I am being subjected to abusive online comments.",
            "Someone shared a harmful post about me publicly.",
            "A person is repeatedly posting insulting material about me.",
            "Someone uploaded content containing offensive statements about me.",
            "A person posted harmful claims about me on social media.",
            "Someone is using online posts to damage my reputation.",
            "I am receiving offensive comments from someone online.",
        ),
    ),
    ComplaintClassifier(
        "Robbery & Snatching",
        "Property taken by force, threat, or sudden snatching",
        (304, 309, 310, 311, 312),
        (
            "snatched", "snatching", "grabbed my phone", "grabbed my bag",
            "pulled my chain", "threatened me and took", "forcefully took",
            "robbed", "robbery", "dacoity", "armed robbery",
        ),
        (
            "A person snatched my phone from my hand.",
            "Someone grabbed my purse and ran away.",
            "Two people threatened me and took my wallet.",
            "A person pulled my gold chain from my neck.",
            "Someone forcefully took my mobile phone on the street.",
            "I was threatened with a weapon and robbed.",
            "A person on a motorcycle snatched my bag.",
            "Someone suddenly grabbed my phone while I was walking.",
            "A group stopped me and took my belongings by force.",
            "My chain was pulled from my neck by a stranger.",
            "Someone threatened me before taking my cash.",
            "A man grabbed my handbag and escaped.",
        ),
    ),

    ComplaintClassifier(
        "Extortion",
        "Demanding property or money through threats or coercion",
        (119, 120, 308),
        (
            "extortion", "demanded money", "pay me or", "blackmail",
            "threatened to expose", "threatened to publish", "ransom demand",
            "give me money", "unless you pay",
        ),
        (
            "Someone threatened to expose my private information unless I paid money.",
            "A person demanded money and threatened to harm me if I refused.",
            "I am being blackmailed for money.",
            "Someone said they would publish my photos unless I paid.",
            "A person is demanding money in exchange for not revealing information.",
            "I received a threat to pay money or face consequences.",
            "Someone demanded cash by threatening me.",
            "A person threatened to leak private material unless I paid.",
            "I am being pressured to pay money because of a threat.",
            "Someone said they would damage my reputation unless I paid.",
        ),
    ),

    ComplaintClassifier(
        "Cheating",
        "Deception used to obtain money, property, or an unfair benefit",
        (318,),
        (
            "cheated", "cheating", "deceived me", "fraudulent promise",
            "false promise", "took my money", "payment after lying",
            "misled me", "deceptive",
        ),
        (
            "Someone deceived me and took my money.",
            "I was cheated after being given a false promise.",
            "A seller lied about a product and kept my payment.",
            "Someone made a false representation to obtain my money.",
            "I paid after being misled about what I would receive.",
            "A person deceived me into handing over property.",
            "Someone promised a service and cheated me after taking payment.",
            "I was tricked by a false business claim.",
            "A person deliberately misled me to get a financial benefit.",
            "Someone cheated me through a dishonest promise.",
        ),
    ),

    ComplaintClassifier(
        "Cheating by Personation",
        "Deception by pretending to be another person or authority",
        (319,),
        (
            "personation", "impersonated", "pretended to be", "posed as",
            "fake bank manager", "fake police officer", "pretending to be my",
            "pretending to work for", "impersonating",
        ),
        (
            "Someone pretended to be my bank manager and took money from me.",
            "A person impersonated a police officer and demanded payment.",
            "Someone posed as my relative and asked me for money.",
            "A caller pretended to work for my bank and deceived me.",
            "Someone pretended to be a government official to obtain money.",
            "A person used a fake identity to trick me into making a payment.",
            "Someone impersonated my friend and asked for money online.",
            "A scammer pretended to be a company representative.",
            "Someone posed as an employee of the bank and obtained my details.",
            "A person pretended to be someone else to deceive me.",
        ),
    ),

    ComplaintClassifier(
        "Criminal Breach of Trust",
        "Property or money entrusted to someone and then dishonestly misused",
        (316,),
        (
            "entrusted", "entrusted money", "misused", "dishonestly used",
            "kept my money", "failed to return", "breach of trust",
            "money I gave",
        ),
        (
            "I gave someone money to keep for me and they used it dishonestly.",
            "My employee was entrusted with cash and kept it.",
            "I handed over property for safekeeping and it was misused.",
            "A person who was holding my money refused to return it and used it.",
            "Someone dishonestly used property I had entrusted to them.",
            "I entrusted a laptop to someone who then disposed of it.",
            "An employee misused money entrusted to them.",
            "Someone retained my property after I gave it to them for a specific purpose.",
        ),
    ),

    ComplaintClassifier(
        "Property Damage",
        "Intentional damage or destruction of property",
        (324, 325, 326, 327),
        (
            "damaged my property", "broke my window", "damaged my car",
            "destroyed", "vandalized", "vandalised", "property damage",
            "set fire to", "damaged the vehicle", "killed my animal",
        ),
        (
            "Someone deliberately damaged my car.",
            "A person broke the windows of my house.",
            "Someone vandalized my shop.",
            "A person intentionally destroyed my property.",
            "Someone damaged my vehicle during an argument.",
            "A person set fire to property belonging to me.",
            "Someone deliberately damaged equipment at my workplace.",
            "A person caused serious damage to my house.",
            "Someone destroyed my belongings on purpose.",
            "A person intentionally damaged a boundary wall.",
        ),
    ),

    ComplaintClassifier(
        "Trespass & House-Breaking",
        "Unauthorised entry onto property, into a house, or breaking into premises",
        (329, 330, 331, 332, 333, 334),
        (
            "entered my house", "entered my home", "broke the lock",
            "broke into", "trespassed", "housebreaking", "house-breaking",
            "entered without permission", "forced the door", "climbed into",
        ),
        (
            "Someone entered my house without permission.",
            "A person broke the lock and entered my home.",
            "Someone broke into my office at night.",
            "A stranger trespassed onto my property.",
            "Someone entered my house by forcing the door open.",
            "A person climbed into my house without permission.",
            "Someone broke a window to enter my shop.",
            "A person entered my property despite being told not to.",
            "Someone broke into a room in my house.",
            "A person entered my premises unlawfully at night.",
        ),
    ),

    ComplaintClassifier(
        "Forgery & False Documents",
        "Creating, altering, or using forged documents or electronic records",
        (335, 336, 337, 338, 339, 340, 341, 342, 343, 344),
        (
            "forged", "forgery", "fake document", "fabricated document",
            "altered certificate", "fake signature", "fake stamp",
            "forged record", "false document", "forged electronic record",
        ),
        (
            "Someone created a forged document using my name.",
            "A person used a fake signature on a document.",
            "Someone fabricated a certificate.",
            "A forged property document was submitted in my name.",
            "Someone altered an official record without permission.",
            "A person used a fake stamp on a document.",
            "Someone created a false electronic record.",
            "A person presented a forged document as genuine.",
            "Someone falsified company accounts.",
            "A forged will was prepared using false information.",
        ),
    ),

    ComplaintClassifier(
        "Criminal Intimidation",
        "Threats intended to cause fear or force a person to act or refrain from acting",
        (351,),
        (
            "threatened me", "threatening me", "threatened to hurt",
            "threatened to kill", "threatened to harm me",
            "if i reported him", "if i report this", "death threat",
            "threat message", "threatened my family", "threatening messages",
            "intimidated",
        ),
        (
            "Someone threatened to hurt me.",
            "I received a death threat.",
            "A person threatened my family.",
            "Someone sent me threatening messages.",
            "A person said they would harm me if I reported the matter.",
            "I was threatened with serious consequences.",
            "Someone threatened to damage my property if I refused to cooperate.",
            "A person intimidated me into staying silent.",
            "I received repeated threats from someone.",
            "Someone threatened me and my family over a dispute.",
        ),
    ),

    ComplaintClassifier(
        "Sexual Offence",
        "Sexual violence, unwanted sexual conduct, harassment, voyeurism, or stalking of a sexual nature",
        (63, 64, 65, 70, 74, 75, 76, 77, 78, 79),
        (
            "rape", "sexual assault", "sexual harassment", "unwanted sexual",
            "touched me", "voyeurism", "stalking", "followed me repeatedly",
            "disrobe", "harassed me sexually",
        ),
        (
            "Someone sexually harassed me.",
            "A person made unwanted sexual comments toward me.",
            "Someone repeatedly followed me in a sexual context.",
            "A person secretly recorded me in a private place.",
            "Someone used criminal force against a woman with sexual intent.",
            "A person threatened me with unwanted sexual conduct.",
            "Someone repeatedly contacted me with sexual messages.",
            "A person tried to disrobe me by force.",
            "Someone watched me without my knowledge in a private situation.",
            "I reported a sexual assault.",
        ),
    ),

    ComplaintClassifier(
        "Kidnapping & Abduction",
        "Taking, inducing, or keeping a person away from lawful custody or for an unlawful purpose",
        (87, 97, 137, 139, 140, 142),
        (
            "kidnapped", "kidnapping", "abducted", "abduction", "held captive",
            "taken against their will", "taken away", "ransom", "forced into a vehicle",
            "kept in confinement",
        ),
        (
            "Someone kidnapped my family member.",
            "A person abducted someone and took them away by force.",
            "My relative was taken against their will.",
            "Someone forced a person into a vehicle and drove away.",
            "A person was held captive and not allowed to leave.",
            "Someone kidnapped a child.",
            "A person was abducted for ransom.",
            "Someone unlawfully kept a kidnapped person in confinement.",
            "A person was taken away without their consent.",
            "Someone kidnapped a person and demanded money for release.",
        ),
    ),

    ComplaintClassifier(
        "Wrongful Restraint & Confinement",
        "Preventing a person from moving freely or unlawfully confining them",
        (126, 127),
        (
            "blocked me", "would not let me leave", "locked me in",
            "confined me", "restrained me", "prevented me from leaving",
            "held me inside", "locked the door", "would not allow me to leave",
        ),
        (
            "Someone blocked my path and would not let me leave.",
            "A person locked me inside a room.",
            "Someone confined me against my will.",
            "A person prevented me from leaving the building.",
            "Someone locked the door and kept me inside.",
            "I was restrained and not allowed to leave.",
            "A person stopped me from moving away.",
            "Someone held me in a room against my wishes.",
        ),
    ),

    ComplaintClassifier(
        "Attempted Murder",
        "Conduct intended to cause death but where the victim survives",
        (109, 110),
        (
            "tried to kill me", "attempted to kill", "attack intended to kill",
            "shot at me", "stabbed me", "poisoned me", "tried to murder",
            "attempted murder", "tried to cause my death",
        ),
        (
            "Someone tried to kill me during an argument.",
            "A person attempted to murder me.",
            "Someone attacked me with the intention of causing my death.",
            "A person shot at me but I survived.",
            "Someone stabbed me during an attack intended to kill me.",
            "A person tried to cause my death deliberately.",
            "Someone attempted to murder a family member.",
            "A person attacked me with an apparent intention to kill.",
        ),
    ),

    ComplaintClassifier(
        "Public Mischief & Harmful Statements",
        "False or harmful statements likely to cause public alarm or disorder",
        (353,),
        (
            "false information", "public alarm", "rumour", "rumor",
            "spread false news", "caused panic", "public mischief",
            "false statement to the public",
        ),
        (
            "Someone spread false information that caused public panic.",
            "A person circulated a false rumour likely to cause alarm.",
            "Someone knowingly shared false information with the public.",
            "A person spread a false statement that could disturb public order.",
            "Someone posted a false rumour that caused panic in the community.",
            "A person circulated misleading information likely to create public fear.",
        ),
    ),

    ComplaintClassifier(
        "Dowry Death & Matrimonial Cruelty",
        "Dowry-related death or cruelty by a husband or his relatives",
        (80, 85),
        (
            "dowry", "dowry demand", "dowry harassment", "husband's family",
            "cruelty by husband", "in-laws demanded", "harassed for dowry",
        ),
        (
            "My husband's family demanded dowry and subjected me to cruelty.",
            "My in-laws harassed me because of a dowry demand.",
            "My husband and his relatives repeatedly demanded more dowry.",
            "I was subjected to cruelty by my husband over dowry.",
            "My family was pressured for additional dowry after marriage.",
            "A woman died after sustained dowry-related harassment by her husband and relatives.",
        ),
    ),

    ComplaintClassifier(
        "Offence Against Public Servant",
        "Assault, obstruction, or other specified unlawful conduct involving a public servant",
        (121, 132, 195, 204),
        (
            "public servant", "police officer", "government officer", "obstructed the officer",
            "attacked a police officer", "impersonated a police officer",
            "prevented the officer", "obstructed public servant",
        ),
        (
            "Someone assaulted a police officer while the officer was performing official duties.",
            "A person obstructed a public servant from carrying out lawful duties.",
            "Someone threatened a government officer during official work.",
            "A person impersonated a public servant.",
            "Someone used force to stop a police officer from performing their duty.",
            "A person attacked an officer while the officer was working.",
        ),
    ),

)


# ---------------------------------------------------------------------------
# CLASSIFIER
# ---------------------------------------------------------------------------

class TopicClassifier:
    """
    First-stage complaint classifier.

    The classifier is deliberately a routing layer rather than a legal
    conclusion. It predicts the most likely plain-language complaint topic
    and supplies the corresponding BNS search space.
    """

    def __init__(
        self,
        approved_examples: tuple[tuple[str, str], ...] = (),
        training_data_path: str | None = None,
    ) -> None:

        self.classifiers = CLASSIFIERS
        self._by_name = {item.name: item for item in CLASSIFIERS}

        # Complaint-classifier training data is maintained separately from
        # the held-out test set. The application never trains on complaint_test.csv.
        from pathlib import Path
        import csv

        default_path = Path(__file__).resolve().parents[1] / 'data' / 'complaint_training.csv'
        path = Path(training_data_path) if training_data_path else default_path

        training_texts: list[str] = []
        labels: list[str] = []

        if path.exists():
            with path.open('r', encoding='utf-8', newline='') as f:
                for row in csv.DictReader(f):
                    text = (row.get('complaint') or '').strip()
                    label = (row.get('category') or '').strip()
                    if text and label in self._by_name:
                        training_texts.append(text)
                        labels.append(label)

        # Fail safely if a user removes the generated CSV: use the embedded
        # starter examples, but never read the held-out complaint test file.
        if not training_texts:
            for item in CLASSIFIERS:
                for example in item.examples:
                    training_texts.append(example)
                    labels.append(item.name)

        for text, label in approved_examples:
            text = text.strip()
            if text and label in self._by_name:
                # Reviewed examples are higher-quality than the starter data.
                for _ in range(3):
                    training_texts.append(text)
                    labels.append(label)

        # Word + character features make the classifier tolerant of paraphrases,
        # spelling variations, and real-world complaint wording.
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            max_features=20_000,
        )

        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
            max_features=30_000,
        )

        word_matrix = self.word_vectorizer.fit_transform(training_texts)
        char_matrix = self.char_vectorizer.fit_transform(training_texts)

        from scipy.sparse import hstack

        training_matrix = hstack([word_matrix, char_matrix], format='csr')

        self.model = LogisticRegression(
            class_weight='balanced',
            max_iter=2_000,
            C=2.0,
            random_state=42,
        )
        self.model.fit(training_matrix, labels)

        self._prototype_names = [item.name for item in CLASSIFIERS]
        self._prototype_word = []
        self._prototype_char = []
        for name in self._prototype_names:
            idx = [i for i, label in enumerate(labels) if label == name]
            self._prototype_word.append(word_matrix[idx].mean(axis=0))
            self._prototype_char.append(char_matrix[idx].mean(axis=0))
        self._prototype_word = __import__('numpy').vstack([m.A for m in self._prototype_word])
        self._prototype_char = __import__('numpy').vstack([m.A for m in self._prototype_char])

    # ------------------------------------------------------------------
    # Keyword signal
    # ------------------------------------------------------------------

    def _keyword_scores(self, text: str) -> dict[str, float]:
        """
        Calculate a soft keyword signal.

        Exact keyword matches are useful as an explainable signal, but
        they are intentionally given much less influence than the model.
        """

        normalised = re.sub(r"\s+", " ", text.lower()).strip()

        scores: dict[str, float] = {}

        for item in self.classifiers:
            score = 0.0

            for keyword in item.keywords:
                pattern = rf"(?<!\w){re.escape(keyword.lower())}(?!\w)"

                if re.search(pattern, normalised):
                    # Longer phrases are more informative than generic words.
                    words = len(keyword.split())

                    if words >= 3:
                        score += 1.5
                    elif words == 2:
                        score += 1.25
                    else:
                        score += 1.0

            scores[item.name] = score

        total = sum(scores.values())

        if total == 0:
            return {
                name: 0.0
                for name in scores
            }

        return {
            name: score / total
            for name, score in scores.items()
        }

    # ------------------------------------------------------------------
    # ML signal
    # ------------------------------------------------------------------

    def _model_scores_from_matrices(self, word_matrix, char_matrix) -> dict[str, float]:
        from scipy.sparse import hstack
        matrix = hstack([word_matrix, char_matrix], format='csr')
        probabilities = self.model.predict_proba(matrix)[0]
        return dict(zip(self.model.classes_, probabilities))

    def _prototype_scores_from_matrices(self, word_matrix, char_matrix) -> dict[str, float]:
        word_sim = cosine_similarity(word_matrix, self._prototype_word).ravel()
        char_sim = cosine_similarity(char_matrix, self._prototype_char).ravel()
        combined = 0.65 * word_sim + 0.35 * char_sim
        return {name: max(0.0, float(score)) for name, score in zip(self._prototype_names, combined)}

    # ------------------------------------------------------------------
    # Final probabilities
    # ------------------------------------------------------------------

    def probabilities(self, text: str) -> dict[str, float]:
        """
        Return probabilities for every complaint category.

        Model signal = 75%
        Keyword signal = 25%

        The keyword component exists mainly for interpretability and
        robustness on very short complaints.
        """

        if not text or not text.strip():
            return {
                item.name: 0.0
                for item in self.classifiers
            }

        # Transform once and reuse the matrices for both the classifier and
        # prototype retrieval. This avoids four vectorizer passes per query.
        word_matrix = self.word_vectorizer.transform([text])
        char_matrix = self.char_vectorizer.transform([text])
        model_scores = self._model_scores_from_matrices(word_matrix, char_matrix)
        keyword_scores = self._keyword_scores(text)
        prototype_scores = self._prototype_scores_from_matrices(word_matrix, char_matrix)

        names = [item.name for item in self.classifiers]
        blended = {}

        # Retrieval + explicit keywords make the multiclass probabilities more
        # decisive without discarding the statistical model.
        for name in names:
            model_score = model_scores.get(name, 0.0)
            keyword_score = keyword_scores.get(name, 0.0)
            prototype_score = prototype_scores.get(name, 0.0)

            blended[name] = (
                0.55 * model_score
                + 0.20 * keyword_score
                + 0.25 * prototype_score
            )

        # When a distinctive phrase belongs strongly to one category, reward
        # that category before normalization. This is safer than blindly
        # lowering the uncertainty threshold.
        for name, score in keyword_scores.items():
            if score >= 0.22:
                blended[name] *= 1.0 + min(0.50, score)

        total = sum(blended.values())

        if total <= 0:
            return {
                name: 0.0
                for name in names
            }

        return {
            name: score / total
            for name, score in blended.items()
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        text: str,
        threshold: float = 0.34,
        margin: float = 0.055,
    ) -> dict:
        """
        Predict the most likely category.

        Returns an explicit 'uncertain' result when:
        - confidence is below threshold, OR
        - the top two categories are too close.

        This is important because real complaints frequently overlap.
        """

        probabilities = self.probabilities(text)

        ranked = sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_name, top_score = ranked[0]

        second_score = (
            ranked[1][1]
            if len(ranked) > 1
            else 0.0
        )

        confidence_margin = top_score - second_score

        top_keywords = self._keyword_scores(text).get(top_name, 0.0)
        strong_phrase = top_keywords >= 0.22

        if (
            top_score < threshold
            or (confidence_margin < margin and not strong_phrase)
        ):
            return {
                "category": None,
                "confidence": top_score,
                "margin": confidence_margin,
                "status": "uncertain",
                "probabilities": dict(ranked),
            }

        classifier = self.classifier(top_name)

        return {
            "category": top_name,
            "confidence": top_score,
            "margin": confidence_margin,
            "status": "classified",
            "section_numbers": classifier.section_numbers,
            "description": classifier.description,
            "probabilities": dict(ranked),
        }

    # ------------------------------------------------------------------
    # Existing API compatibility
    # ------------------------------------------------------------------

    def classifier(self, name: str) -> ComplaintClassifier:
        return self._by_name[name]