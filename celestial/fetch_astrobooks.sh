#!/bin/bash
# fetch_astrobooks.sh -- download the 24-book astronomy corpus from GITenberg
# (Project Gutenberg mirrored as GitHub repos). Creates ./astrobooks/*.txt
# named by Gutenberg id. The 4-book star-guide corpus used in the paper is
# ids 20769, 36741, 68391, 57091; the rest populate the appendix table.
mkdir -p astrobooks
repos=(
 "A-Field-Book-of-the-Stars_20769"
 "Astronomy-with-an-Opera-glasswith-the-Simplest-of-Optical-Instruments_36741"
 "Round-the-year-with-the-stars-The-chief-beauties-of-the-starry-heavens-as-seen-with-the-naked__68391"
 "Astronomy-for-Young-Australians_57091"
 "Pleasures-of-the-telescopeReaders_28752"
 "Half-Hours-with-the-StarsA-Plain-and-Easy-Guide-to-the-Knowledge-of-the-Constellations_23300"
 "Half-hours-with-the-Telescope--13-Being-a-Popular-Guide-to-the-Use-of-the-Telescope-as-a-Mean__16767"
 "Astronomy-for-Amateurs_25267"
 "Astronomy-for-Young-Folks_45112"
 "Recreations-in-AstronomyWith-Directions-for-Practical-Experiments-and-Telescopic-Work_15620"
 "Star-land-Being-Talks-With-Young-People-About-the-Wonders-of-the-Heavens_60318"
 "The-Story-of-the-Heavens_27378"
 "The-Heavens-Above-A-Popular-Handbook-of-Astronomy_58810"
 "Stargazing-Past-and-Present_53172"
 "Stories-of-Starland_54913"
 "The-Children-s-Book-of-Stars_28853"
 "Astronomical-MythsBased-on-Flammarions-s-History-of-the-Heavens_36495"
 "The-Book-of-Stars-Being-a-Simple-Explanation-of-the-Stars-and-Their-Uses-to-Boy-Life_67234"
 "The-New-Heavens_19395"
 "A-Text-Book-of-Astronomy_34834"
 "Letters-on-AstronomyMost-Eminent-Astronomers_40240"
 "Myths-and-Marvels-of-Astronomy_26556"
 "Ancient-calendars-and-constellations_70052"
 "Popular-lessons-in-astronomy_71943"
)
for repo in "${repos[@]}"; do
  id=$(echo "$repo" | grep -oE '[0-9]+$')
  [ -f "astrobooks/$id.txt" ] && { echo "have $id"; continue; }
  for fname in "$id.txt" "$id-8.txt" "$id-0.txt"; do
    code=$(curl -s -o "astrobooks/$id.txt" -w "%{http_code}" \
      "https://raw.githubusercontent.com/GITenberg/$repo/master/$fname")
    if [ "$code" = "200" ]; then echo "$id OK ($fname)"; break; fi
    rm -f "astrobooks/$id.txt"
  done
done
echo "done: $(ls astrobooks | wc -l) books, $(cat astrobooks/*.txt | wc -w) words"