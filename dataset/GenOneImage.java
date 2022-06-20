//import javafx.scene.text.Font;

import org.openscience.cdk.depict.Abbreviations;
import org.openscience.cdk.depict.Depiction;
import org.openscience.cdk.depict.DepictionGenerator;
import org.openscience.cdk.exception.CDKException;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.smiles.SmilesParser;

import java.awt.*;
import java.io.IOException;

//import org.openscience.cdk.renderer.font.AWTFontManager;

/**
 * @author jixiuyi
 * @date 2019-12-27 12:52
 */
public class GenOneImage {
    public static void main(String[] args) throws CDKException, IOException {
        IChemObjectBuilder bldr   = SilentChemObjectBuilder.getInstance();
        Abbreviations abrv = new Abbreviations();
        SmilesParser smipar = new SmilesParser(bldr);
        String smiles="CNC1=NC(=NC(=C1)C2=CC=CC=C2)N3CCC(CC3)C(=O)NCC4=CC=CC=C4C(F)(F)F";
        if (smiles.indexOf("[*]")!=-1){
            String[] str = { "[A]", "[W]"};
            int random1 = (int) ( Math.random () * 2 );
            System.out.println (str[random1]);
            int random2 =(random1+1)%2;
            System.out.println (str[random2]);
            smiles=smiles.replace("[*]",str[random1]);
        }

        if (smiles.indexOf("[F,Cl,Br,I]")!=-1){
            smiles=smiles.replace("F,Cl,Br,I","X");
        }
        System.out.println (smiles);
        IAtomContainer mol = smipar.parseSmiles(smiles);
        //mol.setProperty(CDKConstants.TITLE, "caffeine");

        DepictionGenerator dptgen = new DepictionGenerator();
        DepictionGenerator dpt =dptgen.withZoom(2.0)
                .withMolTitle().withTitleColor(Color.DARK_GRAY).withBackgroundColor(Color.WHITE)
                .withAromaticDisplay();

        dpt.depict(mol)
                .writeTo("C:\\Users\\XuZhanPeng\\Desktop\\caffeine1.png");

        dpt.depict(mol)
                .writeTo(Depiction.JPG_FMT,"C:\\Users\\XuZhanPeng\\Desktop\\mol.jpg");


    }
}