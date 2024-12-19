def entity_linker(nlp, text, linker_name="umls_linker"):
    doc = nlp(text)
    linker = nlp.get_pipe(linker_name)
    return [(ent.text, 
             ent.label_, 
             ent._.kb_ents[0][0] if ent._.kb_ents != [] else None, 
             linker.kb.cui_to_entity[ent._.kb_ents[0][0]].canonical_name\
        if ent._.kb_ents != [] else None) for ent in doc.ents]