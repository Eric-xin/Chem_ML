## Attributes Generators

## General Information

1. Stoichiometric attributes that depend only on the fractions of elements present and not what those elements actually are. These include the number of elements present in the compound and several Lp norms of the fractions.  
2. Elemental property statistics, which are defined as the mean, mean absolute deviation, range, minimum, maximum and mode of 22 different elemental properties. This category includes attributes such as the maximum row on periodic table, average atomic number and the range of atomic radii between all elements present in the material.  
3. Electronic structure attributes, which are the average fraction of electrons from the s, p, d and f valence shells between all present elements. These are identical to the attributes used by Meredig et al.
4. Ionic compound attributes that include whether it is possible to form an ionic compound assuming all elements are present in a single oxidation state, and two adaptations of the fractional ‘ionic character’ of a compound based on an electronegativitybased measure.

## Example Usage

```python
from generator.stoichiometric import StoichiometricAttributeGenerator

# Generate the attributes
attribute_generator = StoichiometricAttributeGenerator()
attributes = attribute_generator.generate_features([CompositionEntry("Na2CO3")])
```