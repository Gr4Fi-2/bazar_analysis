# BAZAAR_KNOWLEDGE.md

## Zweck

Diese Datei ist die Gameplay- und Analyse-Wissensbasis fuer The Bazaar in diesem Repo. Sie soll verhindern, dass Analysebegriffe wie `Scaling`, `Engine`, `Enabler` oder `Payoff` frei geraten werden.

Bei Gameplay-, Meta-, Build-, Engine-, Scaling-, Vendor-, Event- oder Statusfragen zuerst diese Datei lesen, dann lokale Daten und BazaarDB pruefen.

## Quellen Und Rangfolge

1. Lokale Run-Daten in `data/db/bazar_analysis.duckdb`, `runs.board_cards_json`, `runs.skill_cards_json`, `extracted_board_items`, `extracted_skills` und `data/exports/*` sind fuer empirische Fragen massgeblich.
2. BazaarDB ist fuer aktuelle Card-Texte, Tags, Suchsyntax, Card-Seiten, Enchantments, Merchants, Trainers, Monsters, Events und Patch-Kontext die wichtigste Webquelle.
3. BazaarDB Search Docs: `https://bazaardb.gg/docs` beschreibt Filter wie `n:`, `t:`, `s:`, `r:`, `o:`, `c:`, `a:`, `cc:`, `qo:`, `ro:`, `e:`, `et:`, `eo:`, `tc:` und `d:`.
4. BazaarDB Card Pages: `https://bazaardb.gg/card/<id>/<slug>` liefern aktuelle Tooltips, Tags, Enchantments, Where-to-Find, Merchants, History und Deep Mechanics, wenn vorhanden.
5. BazaarDB Kategorien: `/?c=items`, `/?c=skills`, `/?c=merchants`, `/?c=trainers`, `/?c=monsters`, `/?c=events` und Suchseiten mit `t:`/`o:`-Filtern sind crawlbare Quellen.
6. `https://thebazaar.wiki.gg/` ist sekundaerer Kontext. Die Wiki-Hauptseite markiert sich selbst als nicht vollstaendig aktuell; nutze sie fuer Strukturbegriffe, nicht als letzte Wahrheit gegen BazaarDB.
7. Offizielle Seiten koennen JavaScript-lastig sein und sind fuer maschinelle Extraktion oft weniger nuetzlich als BazaarDB.

Patch-Risiko: Live-BazaarDB kann neuer sein als lokale Runs. Bei Auswertungen immer notieren, ob Daten aus lokaler DB oder live Card-Texten stammen.

## Datenquellen Im Repo

- `runs.board_cards_json`: source-first Board-Karten aus BazaarDB-Run-Payloads, mit `title`, `base_id`, `tier`, `enchantment`, `slot_position`.
- `runs.skill_cards_json`: source-first Skill-Karten aus Run-Payloads.
- `reference_items` und `reference_skills`: lokale Referenzen mit BazaarDB IDs, Namen, Slugs, URLs, Icons und teilweise Metadata.
- `data/reference/html/`: gecachte BazaarDB Listen-/Suchseiten, vor allem nuetzlich fuer Namen, Tags und Listenkontext.
- `data/raw/runs_html/`: gecachte Run-Seiten; meist gut fuer Boards/Meta-Beschreibung, nicht zwingend fuer volle Card-Regeltexte.
- `data/exports/summary_*_by_hero.csv`: bevorzugt fuer hero-spezifische Meta-Fragen.

## BazaarDB Suchmuster

- `t:item`, `t:skill`, `t:merchant`, `t:monster`, `t:eventencounter`, `t:combatencounter` suchen nach Hauptkategorien oder Tags.
- `t:<hero>` filtert nach Helden wie `vanessa`, `pygmalien`, `dooley`, `mak`, `stelle`, `jules`, `karnok`.
- `t:burn`, `t:poison`, `t:regen`, `t:shield`, `t:haste`, `t:charge`, `t:freeze`, `t:slow`, `t:crit`, `t:damage`, `t:health`, `t:value`, `t:gold`, `t:income`, `t:ammo`, `t:cooldown` finden direkte Mechanik-Tags.
- `t:*reference` Tags wie `burnreference`, `hastereference`, `healthreference` bedeuten, dass eine Karte auf diese Mechanik Bezug nimmt, nicht zwingend, dass sie sie selbst ausfuehrt.
- `o:"when you use"`, `o:"this gains"`, `o:"gain max health"`, `o:"for the fight"`, `o:"charge this"`, `o:"permanently"` sind wichtige Tooltip-Suchen fuer Engine-Fragen.
- `d=5`, `d>=10` filtert Encounters nach Day, wenn BazaarDB Day-Daten indexiert.
- `e:all`, `e:fiery`, `eo:"when you use"` sind wichtig fuer Enchantment-Fragen.

## Grundbegriffe

- Hero: spielbarer Charakter mit eigenem Item-/Skill-Pool. Im aktuellen lokalen Datenstand: `Vanessa`, `Karnok`, `Stelle`, `Dooley`, `Pygmalien`, `Mak`, `Jules`.
- Board: aktive Item-Anordnung fuer Fights. Slot-Position und Adjacent-Effekte sind wichtig.
- Stash/Storage: nicht aktive Items; in der lokalen Run-DB normalerweise nicht als finale Board-Praesenz enthalten.
- Item: aktive Karte mit Groesse, Tier, Typen/Tags, ggf. Cooldown, Ammo, Crit Chance und Tooltips.
- Skill: passive Faehigkeit, oft durch Level-Up, Trainer, Monster oder Events.
- Encounter: Run-Knoten wie Monster, Merchant, Trainer, Event oder Player Fight.
- Merchant/Vendor: Encounter, der Items/Services verkauft. BazaarDB Card-Seiten listen haeufig Where-to-Find/Merchants.
- Trainer/Skill Merchant: Encounter fuer Skills; Wiki nennt day-only und level-up-only Skill Merchants als getrennte Gruppen.
- Monster: PvE Fight. Wiki-Kontext: dritte Encounter eines Days ist ein PvE Monster Fight mit drei Auswahloptionen; Rewards umfassen Gold, XP und eines der Monster-Items/-Skills. Gegen BazaarDB pruefen, falls exakte aktuelle Regeln relevant sind.
- Event: Encounter mit Auswahl/Reward/Transformation/Upgrade/Gold/Health/etc. BazaarDB `d:` und Eventseiten/Wiki-Kategorien fuer Day-Gates pruefen.
- Tier/Quality: Bronze, Silver, Gold, Diamond, Legendary. Upgrades erhoehen meist Effekte und Sell Value, aber nicht zwingend jeden Effekt.
- Size: Small, Medium, Large. Wiki-Kontext: Small belegt 1 Slot, Medium 2 Slots, Large 3 Slots.
- Type/Tag: z.B. Weapon, Tool, Food, Property, Vehicle, Friend, Tech, Relic, Potion, Reagent, Aquatic, Dinosaur, Dragon, Drone, Toy, Trap, Loot, Ray, Apparel.
- Enchantment: Modifikation wie Heavy, Golden, Icy, Turbo, Shielded, Restorative, Toxic, Fiery, Shiny, Deadly, Radiant, Obsidian, Mossy. BazaarDB Card-Seiten sind hier massgeblich.
- Quest: Card-spezifische Bedingung/Reward. BazaarDB `qo:`, `ro:`, `rt:` durchsuchen.

## Combat-Mechaniken

Diese Kurzdefinitionen sind Analysehilfen. Fuer exakte Zahlen immer Card-Tooltip und Patch pruefen.

- Cooldown: Zeit bis ein Item ausloest. BazaarDB `c:` filtert Cooldown; Cards ohne Cooldown gelten dort als `0`.
- Charge: schiebt ein Item auf seinem Cooldown um eine angegebene Zeit voran. Wichtig fuer Loops, wenn dadurch Gain- oder Status-Trigger haeufiger ausloesen.
- Haste: temporare Beschleunigung von Item-Aktivierungen. Exakte Formel/Dauer aus Card-Text pruefen. Haste allein ist Enabler/Tempo, kein Scaling.
- Slow: temporaere Verlangsamung/Verzoegerung von Item-Aktivierungen. Slow allein ist Control, kein Scaling.
- Freeze: verhindert/verzoegert ein Item fuer eine Dauer. Freeze allein ist Control, kein Scaling.
- Ammo: begrenzte Nutzungen/Munition fuer ein Item; Reload erhoeht/erneuert Ammo.
- Multicast: zusaetzliche Ausloesungen eines Items bei Nutzung. Multicast kann Payoff oder Enabler sein.
- Crit/Crit Chance: Chance auf kritische Ausloesung; Crit kann Triggerbedingung fuer andere Effekte sein.
- Damage: direkter Schaden.
- Shield: absorbierender Schutz. Shield kann statischer Wert, wachsender Wert, Trigger oder Payoff sein.
- Heal: stellt Health wieder her.
- Regen: wiederkehrende Heilung/Regeneration; haeufig als Scaling- oder Health-Engine relevant.
- Burn: Schaden-ueber-Zeit/Status-Schaden; genaue Tickregeln bei Bedarf extern verifizieren. Burn kann Trigger, Payoff oder Scaling-Wert sein.
- Poison: Schaden-ueber-Zeit/Status-Schaden; genaue Tickregeln bei Bedarf extern verifizieren. Poison kann Trigger, Payoff oder Scaling-Wert sein.
- Rage/Enrage: Karnok-nahe Mechanik; Rage/Enrage-Text immer exakt lesen. Nicht automatisch als Scaling einstufen, wenn nur Status konvertiert wird.
- Lifesteal: Damage heilt den Spieler, sofern der konkrete Text das sagt.
- Destroy: entfernt/zerstoert Items im Fight oder dauerhaft, je nach Text.
- Transform: wandelt Item/Status/Typ um; kann Enabler oder Engine sein, wenn dadurch Gain-Trigger entstehen.
- Flying: Status/Tag-Zustand, besonders Stelle-relevant. `start Flying` und `stop Flying` sind haeufig Trigger.
- Heated/Chilled: Jules-/Food-nahe Zustandsreferenzen. Entscheidend ist, ob der Text nur Zustand setzt oder daraus Werte wachsen laesst.

## Trigger-Sprache

- `Every X sec.`: aktiver Cooldown-Trigger.
- `Passive:`: kontinuierlicher oder bedingter Effekt ohne eigene Aktivierung.
- `At the start of each fight`: Fight-Start-Trigger; haeufig statisch oder einmalig.
- `At the start of each day`: Day-Start-Trigger; kann permanent/economy relevant sein.
- `When you use ...`: Trigger bei Nutzung einer Karte/Typ-Gruppe.
- `When you Crit/Burn/Freeze/Slow/Poison/Shield/Heal/Regen`: Status-/Outcome-Trigger.
- `When you buy/sell/win/level up`: Run-/Economy-/Progression-Trigger.
- `for the fight`: nur fuer aktuellen Fight, kann trotzdem Combat-Scaling sein.
- `permanently`: bleibt ueber Fights/Days erhalten; wichtig fuer Run-Scaling.

## Analyse-Taxonomie

- Source: Karte/Skill, der den eigentlichen Effekt erzeugt.
- Enabler: Karte/Skill, der die Source haeufiger oder frueher ausloest, aber selbst nicht die wachsende Zahl erzeugt.
- Payoff: Karte/Skill, der von einem skalierten Wert profitiert.
- Converter: wandelt Wert A in Wert B um, z.B. Max Health zu Damage. Converter ist nur Scaling, wenn A oder B dynamisch waechst.
- Static Support: starke Aura oder Start-of-fight Effekt ohne wachsende Zahl.
- Feedback Loop: A triggert B, B triggert A oder beschleunigt A. Nur als Scaling zaehlen, wenn ein Wert/Frequenz dadurch waechst.
- Economy Engine: Gold, Value, Income, Buy/Sell, Merchant-/Day-Effekte.
- Tempo Engine: Haste, Charge, Cooldown-Reduktion, Multicast, Reload.
- Status Engine: Burn, Poison, Freeze, Slow, Regen, Shield, Heal, Rage etc. als wiederholte Trigger.
- Defensive Engine: Health, Shield, Heal, Regen, Damage Prevention.
- Damage Engine: Damage, Burn, Poison, Crit, Multicast, Ammo/Reload.

## Scaling-Regeln

Als Scaling Source zaehlt sicher:

- `this gains +X`
- `items gain +X`
- `adjacent items gain +X`
- `gain Max Health`
- `permanently gains`
- `for the fight` mit wachsender Zahl
- dynamische Cooldown-Reduktion wie `reduced by X for each Y`
- wiederholtes Value/Gold/Income/Health/Damage/Shield/Heal/Regen/Burn/Poison/Crit-Wachstum

Als Enabler, aber nicht automatisch als Scaling Source zaehlt:

- Haste ohne Gain-Effekt
- Charge ohne Gain-Effekt
- Freeze/Slow/Burn/Poison als reine Ausloeser ohne eigenen Zuwachs
- Start-of-fight Haste
- Reload ohne Ammo-/Value-/Damage-Growth
- Multicast ohne wachsenden Wert

Nicht als Scaling Source zaehlt:

- reine `equal to Max Health`-, `equal to Value`-, `equal to Gold`-Converter, wenn der Referenzwert nicht durch die Engine waechst
- statische Adjacent-Aura
- statische Cooldown-Aura
- reine Typ-/Tag-Ergaenzung
- hohe Cooccurrence oder hohe 10-Win-Rate ohne Regeltext-Beleg

Beispiele:

- `Spice Rack` ist Scaling Source: Adjacent items gain Crit Chance; when you Crit with adjacent item, your items gain einen Wert. `Instant Noodles` ist meist Enabler/Payoff, nicht die Source.
- `Pasta + Farmer's Market` ist Regen/Health-Engine: Pasta gewinnt Max Health ueber Regen; Farmer's Market bufft Regen und kann Health/Economy staerken.
- `Grill` ist Combat-Scaling Source nur ueber `this gains Burn for the fight`; ohne genug Food/Heated-Trigger langsam.
- `Hidden Lake` ist statischer Support, nicht Scaling Source. Es kann eine andere Source wie `Runic Claymore` unterstuetzen.
- `Holsters` ist Start-of-fight/Tempo-Enabler, nicht Scaling Source.
- `Trail Markers` ist statische Cooldown-Aenderung, nicht Scaling Source.
- `Flame Sigil` ist eher Rage/Burn-Feedback-Converter. Als reine Scaling Source nur zaehlen, wenn ein dynamisch wachsender Wert in der konkreten Engine belegt ist.

## Run- Und Day-Kontext

- Lokale Analyse nutzt finale BazaarDB-Run-Payloads, nicht vollstaendige Shop-/Day-Verlaeufe.
- Finale Board-Praesenz belegt, dass eine Engine im Endzustand vorhanden war, aber nicht, wann sie gefunden wurde oder ob sie kausal gewonnen hat.
- Wiki-Kontext: ein Day enthaelt Encounters, darunter regelmaessig PvE Monster; die dritte Encounter eines Days wird dort als Monster Fight beschrieben. Fuer exakte aktuelle Day-Ablaeufe BazaarDB Encounter-/Day-Daten (`d:`) und aktuelle Quellen pruefen.
- Monster Rewards: Wiki beschreibt Gold, XP und eines der Monster-Items/-Skills. Aktuelle Reward-Pools via BazaarDB/Monster-Seiten pruefen.
- Level-Up und Trainer/Skill Merchants koennen Skills und Upgrades anbieten; lokale finale Runs enthalten Skills, aber nicht zwingend Auswahlpfad.
- Prestige/Crown/Run-Wins in diesem Repo sind Analysefelder; `run_victory_tier` wird im Code abgeleitet und ist nicht automatisch identisch mit jeder Ingame-Benennung.

## Vendors, Trainers, Events, Monsters

Fuer aktuelle Crawls bevorzugt BazaarDB:

- Merchants: `https://bazaardb.gg/?c=merchants` und Card-Seiten in `Where to Find`.
- Trainers: `https://bazaardb.gg/?c=trainers` und Suchfilter `t:skill`, `t:<hero>`.
- Monsters: `https://bazaardb.gg/?c=monsters`, `t:combatencounter`, Day-Filter `d:`.
- Events: `https://bazaardb.gg/?c=events`, `t:eventencounter`, Day-Filter `d:`.
- Sitemap: `https://bazaardb.gg/sitemap.xml` fuer breite Card-/Encounter-Discovery.

Wiki-Kontext nennt Merchant-Beispiele wie Aero, Aila, Ande, Barkun, Chronos, Cobweb, Colt, Curio, Eli, Flex, Freiya, Gaseo, Goldie, Hef, Herma, Jay Jay, Kev's Armory, Kina, Knightshade, Luxe, Midsworth, Mittel, Mr. Morland, Nautica, Orion, Pol, Prospero, Quixel, Serafina, Silvia, Tatiana, Tinker, Tok's Clocks und Valpak. Nutze das nur als Namensliste/Startpunkt; aktuelle Pools aus BazaarDB ziehen.

Wiki-Kontext nennt Skill-Merchants day-only wie Grandmaster, Nufu und Zurphin's Safari sowie level-up-only wie Adira, Argenta, Bjorn, C4, Cymon, Fortis, Malafang, Mr. Tuskari, Old Zane, Orlin, Pip, Professor Riggle, Regenald, Ryukon, Slohmor Lumbra, Vermir, Zara und Zosima. Auch hier aktuelle Pools ueber BazaarDB pruefen.

## Analyse-Workflow Fuer Meta-/Engine-Fragen

1. Frage in Begriffe uebersetzen: Source, Enabler, Payoff, Converter, Static Support, Tempo, Economy, Status, Defensive, Damage.
2. Mechanik aus Regeltext belegen: lokale Referenz, BazaarDB Card Page, `o:`/`t:` Suche, ggf. Wiki nur als Kontext.
3. Statik ausschliessen: keine Engine nur wegen `equal to`, Start-of-fight, statischer Aura oder hoher Winrate.
4. Finale Board-/Skill-Praesenz aus lokaler DB messen.
5. Hero-spezifisch auswerten, nicht globale Cooccurrence als Meta interpretieren.
6. Sample Size, 10-Win-Rate und Lift gegen Hero-Baseline berichten.
7. Unsicherheiten explizit markieren: Patch-Mismatch, geringe Samples, fehlender Day-Verlauf, nur finale Boards.

## Standard-Caveats

- Cooccurrence ist kein Kausalbeweis.
- Hohe Winrate ist kein Mechanikbeweis.
- Eine starke Karte kann Enabler oder Payoff sein, ohne Source zu sein.
- Eine Karte kann in einer Engine Scaling sein und in einer anderen nur statischer Support.
- Bei unklaren Status- oder Day-Regeln nicht raten; BazaarDB/Wiki/weitere Quelle suchen oder als unbekannt markieren.
- Bei live Webfetch gegen BazaarDB beachten: manche Requests brauchen `curl_cffi`/Impersonation; lokale `webfetch` kann 403 bekommen.
