import jsonlines
import spacy

nlp = spacy.load('en')

counter = 0
with jsonlines.open("reliefweb_corpus_raw_20160331_eng.jsonl") as reader:
    with jsonlines.open("eng_tagged_raw_reliefweb_spacy_full.jsonl", "w") as writer:
        for obj in reader:
            temp_obj = {}
            temp_obj["id"] = obj["id"]
            tag_list = []
            doc = nlp(obj["text"])
            for ent in doc.ents:
                tag_list.append([ent.text, ent.label_])
            temp_obj["tags"] = tag_list

            writer.write(temp_obj)
            counter += 1
            # print "finished tagging " + str(counter) + " / 426016 docs"
            if counter % 10000 == 0:
                with open("check/" + str(counter), "w") as fi:
                    fi.write("ok")
